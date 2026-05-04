import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import rlax

from models import PPOAgent
from envs.jax_complex_hems import JAXComplexHemsEnv

# --- HYPERPARAMETERS ---
NUM_ENVS = 2048
ROLLOUT_STEPS = 24       # One full day
TOTAL_TIMESTEPS = 20_000_000
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

def main():
    print("Initializing AACL-PPO Training Pipeline...")
    env = JAXComplexHemsEnv("data/processed/heeten_complex_hems_data.npz")
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0))

    # AACL Hardware Masking
    allowed_actions = [0, 4, 8] # Discharge, Idle, Charge
    env = env.set_hardware_topology(allowed_actions)

    # Initialize Agent & Optimizer
    rng = jax.random.PRNGKey(42)
    rng, init_key = jax.random.split(rng)
    
    agent = PPOAgent(obs_dim=7, act_dim=env.num_actions, key=init_key)
    optimizer = optax.adam(LR)
    
    # Equinox specific: We only optimize the arrays (weights), not the static functions
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_array))

    # Get Initial State
    rng, reset_key = jax.random.split(rng)
    obs, state = vmap_reset(jax.random.split(reset_key, NUM_ENVS))

    # ==========================================
    # 1. ROLLOUT FUNCTION (Gather Data)
    # ==========================================
    @eqx.filter_jit
    def rollout_trajectory(current_agent, init_state, init_obs, rng_key):
        def step_fn(carry, step_key):
            curr_state, curr_obs = carry
            
            # Forward pass
            logits = jax.vmap(current_agent)(curr_obs)
            masked_logits = logits + jnp.where(env.allowed_actions_mask, 0.0, -1e9)
            values = jax.vmap(current_agent.get_value)(curr_obs).squeeze()
            
            # Sample Action
            actions = jax.random.categorical(step_key, masked_logits)
            log_probs = jax.nn.log_softmax(masked_logits)[jnp.arange(NUM_ENVS), actions]
            
            # Step Env
            next_obs, next_state, rewards, dones = vmap_step(curr_state, actions)
            
            # Store Transition
            transition = (curr_obs, actions, rewards, values, log_probs, dones)
            return (next_state, next_obs), transition

        keys = jax.random.split(rng_key, ROLLOUT_STEPS)
        (final_state, final_obs), transitions = jax.lax.scan(step_fn, (init_state, init_obs), keys)
        return final_state, final_obs, transitions

    # ==========================================
    # 2. PPO LOSS FUNCTION
    # ==========================================
    @eqx.filter_value_and_grad
    def compute_loss(model, obs_b, actions_b, advantages_b, returns_b, old_log_probs_b):
        # Flatten batches: (Steps, Envs, ...) -> (Steps * Envs, ...)
        obs_flat = obs_b.reshape(-1, 7)
        actions_flat = actions_b.reshape(-1)
        adv_flat = advantages_b.reshape(-1)
        ret_flat = returns_b.reshape(-1)
        old_log_probs_flat = old_log_probs_b.reshape(-1)

        # Forward pass on all data
        logits = jax.vmap(model)(obs_flat)
        masked_logits = logits + jnp.where(env.allowed_actions_mask, 0.0, -1e9)
        values = jax.vmap(model.get_value)(obs_flat).squeeze()

        # Calculate Policy Loss (Actor)
        new_log_probs = jax.nn.log_softmax(masked_logits)[jnp.arange(len(actions_flat)), actions_flat]
        ratio = jnp.exp(new_log_probs - old_log_probs_flat)
        
        # RLax handles the clipping math perfectly
        policy_loss = rlax.clipped_surrogate_pg_loss(ratio, adv_flat, CLIP_EPS)

        # Calculate Value Loss (Critic)
        value_loss = jnp.mean(jnp.square(ret_flat - values))

        # Calculate Entropy (Exploration)
        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(masked_logits) * jax.nn.log_softmax(masked_logits), axis=-1))

        # Total Loss
        total_loss = policy_loss + (VF_COEF * value_loss) - (ENT_COEF * entropy)
        return total_loss

    # ==========================================
    # 3. UPDATE LOOP
    # ==========================================
    @eqx.filter_jit
    def update_step(current_agent, current_opt_state, state_b, obs_b, rng_key):
        # Gather Experience
        next_state, next_obs, transitions = rollout_trajectory(current_agent, state_b, obs_b, rng_key)
        obs_seq, actions_seq, rewards_seq, values_seq, log_probs_seq, dones_seq = transitions
        
        # Bootstrap final value for GAE
        final_values = jax.vmap(current_agent.get_value)(next_obs).squeeze()
        
        # Calculate GAE and Returns (using RLax)
        # Note: RLax expects [Time, Batch], which perfectly matches our output from jax.lax.scan!
        discounts_seq = GAMMA * (1.0 - dones_seq)
        advantages_seq = jax.vmap(rlax.truncated_generalized_advantage_estimation, in_axes=(1, 1, None, 1), out_axes=1)(
            rewards_seq, discounts_seq, GAE_LAMBDA, jnp.concatenate([values_seq, final_values[None, :]])
        )
        returns_seq = advantages_seq + values_seq
        
        # Normalize Advantages
        advantages_seq = (advantages_seq - jnp.mean(advantages_seq)) / (jnp.std(advantages_seq) + 1e-8)

        # Compute gradients and update model
        loss, grads = compute_loss(current_agent, obs_seq, actions_seq, advantages_seq, returns_seq, log_probs_seq)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_agent)
        new_agent = eqx.apply_updates(current_agent, updates)

        return new_agent, new_opt_state, next_state, next_obs, loss, jnp.mean(rewards_seq)

    # ==========================================
    # 4. START TRAINING
    # ==========================================
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_STEPS)
    print(f"Starting {num_updates} Training Updates (Total Steps: {TOTAL_TIMESTEPS})...")
    reward_history = []

    for update in range(1, num_updates + 1):
        rng, step_key = jax.random.split(rng)
        agent, opt_state, state, obs, loss, mean_reward = update_step(agent, opt_state, state, obs, step_key)
        
        if update % 20 == 0 or update == 1:
            print(f"Update {update:04d} | Mean Reward: {mean_reward:8.4f} | Loss: {loss:8.4f}")
        reward_history.append(mean_reward)

    print("Training Complete!")

    eqx.tree_serialise_leaves("trained_hems_agent.eqx", agent)
    print("Model weights saved to trained_hems_agent.eqx")

    import matplotlib.pyplot as plt

    # Inside main(), before the loop:

    # Inside the loop:

    # After training completes:
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history)
    plt.title("AACL-PPO Training Performance (Heeten Dataset)")
    plt.xlabel("Update Iteration")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.savefig("training_curve.png")
    print("Training curve saved as training_curve.png")

if __name__ == "__main__":
    main()

