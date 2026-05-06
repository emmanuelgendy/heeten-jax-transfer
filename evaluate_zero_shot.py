import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from models import PPOAgent
from envs.jax_complex_hems import JAXComplexHemsEnv

def run_eval(agent, env, allowed_actions):
    # Set the topology for this specific test
    test_env = env.set_hardware_topology(allowed_actions)
    
    # Reset env (using a fixed key for fair comparison)
    rng = jax.random.PRNGKey(123)
    obs, state = test_env.reset(rng)
    
    rewards = []
    soc_profile = []
    temp_profile = []
    
    # Simulate one full day (24 steps)
    for _ in range(24):
        logits = agent(obs)
        # Apply the zero-shot mask
        masked_logits = logits + jnp.where(test_env.allowed_actions_mask, 0.0, -1e9)
        action = jnp.argmax(masked_logits)
        
        obs, state, reward, done = test_env.step(state, action)
        
        rewards.append(reward)
        soc_profile.append(state.battery_soc)
        temp_profile.append(state.indoor_temp)
        
    return jnp.array(rewards), jnp.array(soc_profile), jnp.array(temp_profile)

def main():
    # 1. Load Environment and Agent
    env = JAXComplexHemsEnv("data/processed/heeten_complex_hems_data.npz")
    agent = PPOAgent(obs_dim=7, act_dim=env.num_actions, key=jax.random.PRNGKey(0))
    
    # Load your trained weights
    try:
        agent = eqx.tree_deserialise_leaves("trained_hems_agent.eqx", agent)
        print("Successfully loaded trained model weights.")
    except Exception as e:
        print("Could not find trained_hems_agent.eqx. Running with random weights for demo.")

    # 2. Run Comparisons
    print("Evaluating Zero-Shot Scenarios...")
    full_r, full_soc, full_temp = run_eval(agent, env, list(range(9)))
    
    # CORRECTED MASKS:
    # 1 (Cool), 4 (Idle), 7 (Heat) -> HVAC Only
    hvac_r, hvac_soc, hvac_temp = run_eval(agent, env, [1, 4, 7]) 
    
    # 3 (Discharge), 4 (Idle), 5 (Charge) -> Batt Only
    batt_r, batt_soc, batt_temp = run_eval(agent, env, [3, 4, 5])

    # 3. Plotting Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(full_temp, label="Full (Batt+HVAC)", linewidth=2)
    ax1.plot(hvac_temp, label="Zero-Shot (HVAC Only)", linestyle="--")
    ax1.plot(batt_temp, label="Zero-Shot (Batt Only)", linestyle=":")
    ax1.axhline(21, color='red', alpha=0.3, label="Setpoint Low")
    ax1.set_ylabel("Indoor Temp (°C)")
    ax1.set_title("Zero-Shot Thermal Stability Comparison")
    ax1.legend()

    ax2.bar(["Full", "HVAC Only", "Batt Only"], 
           [jnp.sum(full_r), jnp.sum(hvac_r), jnp.sum(batt_r)],
           color=['blue', 'green', 'orange'])
    ax2.set_ylabel("Total Daily Reward (Efficiency)")
    ax2.set_title("Zero-Shot Economic Performance")

    plt.tight_layout()
    plt.savefig("zero_shot_comparison.png")
    print("Comparison plot saved as zero_shot_comparison.png")

if __name__ == "__main__":
    main()