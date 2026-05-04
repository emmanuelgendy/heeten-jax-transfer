import jax
import jax.numpy as jnp
import equinox as eqx

from models import PPOAgent
from envs.jax_complex_hems import JAXComplexHemsEnv

def main():
    # ==========================================
    # 1. INITIALIZE AND VECTORIZE ENVIRONMENT
    # ==========================================
    print("Initializing Vectorized Heeten Environment...")
    env = JAXComplexHemsEnv("data/processed/heeten_complex_hems_data.npz")
    
    num_envs = 2048
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0))

    # ==========================================
    # 2. SET THE HARDWARE TOPOLOGY (AACL)
    # ==========================================
    allowed_actions = [0, 4, 8] 
    env = env.set_hardware_topology(allowed_actions)
    print(f"Hardware Topology Set. Allowed Actions: {allowed_actions}")

    # ==========================================
    # 3. GENERATE PRNG KEYS
    # ==========================================
    # JAX requires explicit random keys for everything (reproducibility!)
    rng = jax.random.PRNGKey(42)
    rng, env_key, init_key = jax.random.split(rng, 3)

    # ==========================================
    # 4. INITIALIZE NEURAL NETWORK AGENT
    # ==========================================
    agent = PPOAgent(
        obs_dim=7, 
        act_dim=env.num_actions, 
        key=init_key
    )
    
    # Get the first batch of observations
    reset_keys = jax.random.split(env_key, num_envs)
    obs_batch, state_batch = vmap_reset(reset_keys)
    print(f"Observation batch shape: {obs_batch.shape}") 

    # ==========================================
    # 5. THE JITTED TRAINING LOOP
    # ==========================================
    @eqx.filter_jit
    def rollout_step(current_agent, state, obs, rng_key):
        # 1. Forward pass through the Actor network
        logits = jax.vmap(current_agent)(obs) 
        
        # 2. AACL Masking
        masked_logits = logits + jnp.where(env.allowed_actions_mask, 0.0, -1e9)
        
        # 3. Sample actions probabilistically based on network output
        actions = jax.random.categorical(rng_key, masked_logits)
        
        # 4. Step the environments
        next_obs, next_state, rewards, dones = vmap_step(state, actions)
        
        return next_obs, next_state, rewards, dones
        
    print("Starting JIT Compilation & Neural Network Rollout...")
    for _ in range(5):
        rng, step_key = jax.random.split(rng)
        obs_batch, state_batch, rewards, dones = rollout_step(agent, state_batch, obs_batch, step_key)
    
    print("Successfully ran 5 JAX-vectorized steps with the Neural Network!")
    print(f"Current Mean Reward: {jnp.mean(rewards):.4f}")

if __name__ == "__main__":
    main()