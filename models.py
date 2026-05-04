import jax
import jax.numpy as jnp
import equinox as eqx

class PPOAgent(eqx.Module):
    """A standard Actor-Critic architecture for PPO."""
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP

    def __init__(self, obs_dim: int, act_dim: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        
        # The Actor decides WHICH action to take (Outputs 9 logits)
        self.actor = eqx.nn.MLP(
            in_size=obs_dim, 
            out_size=act_dim, 
            width_size=64, 
            depth=2, 
            activation=jax.nn.relu, 
            key=k1
        )
        
        # The Critic estimates HOW GOOD the current state is (Outputs 1 value)
        self.critic = eqx.nn.MLP(
            in_size=obs_dim, 
            out_size=1, 
            width_size=64, 
            depth=2, 
            activation=jax.nn.relu, 
            key=k2
        )

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Returns the raw action preferences (logits)."""
        return self.actor(obs)
        
    def get_value(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Returns the critic's state-value estimation."""
        return self.critic(obs)