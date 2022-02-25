from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from torch import nn

from utils.net import weight_init, SACEncoder

from utils.env import *

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common import logger
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import CnnPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic

def tie_weights(src, trg):
    assert type(src) == type(trg)
    if isinstance(src, nn.ReLU):
        return
    trg.weight = src.weight
    trg.bias = src.bias

def copy_conv_weights_from(source, target):
    """Tie convolutional layers"""
    # only tie conv layers
    for i in range(4):
        tie_weights(src=source.convs[i], trg=target.convs[i])


class SACv2(SAC):
    def __init__(self, policy, *args, data_aug='crop', input_size = 108, channels=9, critic_update_steps=1,\
     actor_update_freq=2, init_steps=10000, translate_mid_size=100, _init_setup_model=True, replay_buffer_class=None,replay_buffer_kwargs=None,
      **kwargs):
        # rad or not, that is a problem
        # rad: 
        # crop: policy observation_space (84,84)
        # translate: policy observation_space (108,108)
        # no rad: 
        # crop: policy observation_space (84,84)
        # translate: policy observation_space (108,108)
        # so we don't need to chance policy, we just need to change replay_buffer
        self.data_aug = data_aug
        super(SACv2, self).__init__(SACPolicyv2, *args, replay_buffer_class=ReplayBufferv2, \
            _init_setup_model=False, replay_buffer_kwargs={ \
            'data_aug': data_aug, 'input_size': input_size}, **kwargs)
        self.critic_update_steps = critic_update_steps
        self.actor_update_freq = actor_update_freq 
        #self.logger = logger
        self.init_steps=init_steps
        self._setup_model()
        
    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 0.1
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=1e-4, betas=(0.5, 0.999))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


    def train_actor(self, replay_data):
        ent_coef =  self.get_ent_coef()
        self.actor.optimizer.zero_grad()
        
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations, detach_encoder=True)
        log_prob = log_prob.reshape(-1, 1)
        q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi, detach_encoder=True), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

        # Optimize the actor
        
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.item(), log_prob.mean().item(), min_qf_pi.mean().item()

    def train_critic(self, replay_data):
        ent_coef =  self.get_ent_coef()
        
        self.critic.optimizer.zero_grad()
        
        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions, detach_encoder=False)
        # Compute critic loss
        critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        # Optimize the critic
        
        critic_loss.backward()
        # grad_norm_critic = th.nn.utils.clip_grad_norm_(self.critic.parameters(), 100, norm_type=2.0)
        self.critic.optimizer.step()
        return critic_loss.item(), current_q_values[0].mean().item(), \
        target_q_values.mean().item(), replay_data.rewards.mean().item(), next_log_prob.mean().item()

    def get_ent_coef(self):
        if self.ent_coef_optimizer is not None:
            ent_coef = th.exp(self.log_ent_coef.detach())
        else:
            ent_coef = self.ent_coef_tensor
        return ent_coef
    
    def train_ent_coef(self, replay_data):
        # detach encoder ok here
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations, detach_encoder=True)
        
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            
            ent_coef_loss = (th.exp(self.log_ent_coef) * ((-log_prob-self.target_entropy).detach())).mean()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            for param_group in self.ent_coef_optimizer.param_groups:
                assert param_group["lr"]< 1e-4 + 0.000001

        ent_coef = self.get_ent_coef()
        return ent_coef_loss.item(), ent_coef.item()

    def train(self, gradient_steps: int, batch_size: int = 512) -> None:
        # Update optimizers learning rate
        # optimizers = [self.actor.optimizer, self.critic.optimizer]
        
        # if self.ent_coef_optimizer is not None:
        #     optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        # self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # Sample replay buffer
        assert self._vec_normalize_env==None
        # replay_data = self.replay_buffer.sample(batch_size, env=None)

        # We need to sample because `log_std` may have changed between two gradient steps
        # if self.use_sde:
        #     self.actor.reset_noise()

        # Action by the current actor for the sampled state

        for _ in range(self.critic_update_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=None)
            critic_loss, current_q_value, target_q_value, \
                batch_reward, next_log_prob = self.train_critic(replay_data)
            critic_losses.append(critic_loss)
        self.logger.record("train/current_q_value", current_q_value)
        self.logger.record("train/target_q_value", target_q_value)
        self.logger.record("train/batch_reward",batch_reward)
        self.logger.record("train/next_log_prob", next_log_prob)
        # Compute actor loss
        # Mean over all critic networks
        if self._n_updates % self.actor_update_freq == 0:
            # yes it's training
            # NOTE: use the same replay data
            actor_loss, log_prob, min_q = self.train_actor(replay_data)
            actor_losses.append(actor_loss)
            ent_coef_loss, ent_coef = self.train_ent_coef(replay_data)

            ent_coef_losses.append(ent_coef_loss)
            ent_coefs.append(ent_coef)
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/log_prob", log_prob)
            self.logger.record("train/min_q", min_q)
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # Update target networks
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(self.critic.features_extractor.parameters(), self.critic_target.features_extractor.parameters(), 0.05)
            polyak_update(self.critic.qf0.parameters(), self.critic_target.qf0.parameters(), 0.01)
            polyak_update(self.critic.qf1.parameters(), self.critic_target.qf1.parameters(), 0.01)
            
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))


class SACPolicyv2(CnnPolicy):
    
    def __init__(self, observation_space, *args, input_size = 108, channels = 9, **kwargs):
        # here, data_aug determines how to deal with input image, only 'crop' takes effect
        # thus, here we may change data_aug from no_aug to crop
        self.input_size = input_size
        observation_space = gym.spaces.Box(0, 255, (channels, input_size, input_size), np.uint8)
        super(SACPolicyv2, self).__init__(observation_space,*args,**kwargs)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        
        if self.input_size == 84:
            observation = center_crop_vec(observation, 84)
        elif self.input_size == 108:
            observation = center_translate_vec(observation, 108)
        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, state
    
    def make_actor(self, features_extractor=None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actorv2(input_size=self.input_size, **actor_kwargs).to(self.device)

    def make_critic(self, features_extractor = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCriticv2(input_size=self.input_size, **critic_kwargs).to(self.device)

    def _build(self, lr_schedule) -> None:
        self.critic = self.make_critic()
        self.critic.apply(weight_init)

        if self.share_features_extractor:
            self.actor = self.make_actor(features_extractor=None)
            self.actor.apply(weight_init)
            copy_conv_weights_from(self.actor.features_extractor, self.critic.features_extractor)
            actor_parameters = self.actor.parameters()
            self.actor.optimizer = self.optimizer_class(actor_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        critic_parameters = self.critic.parameters()
        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)

        # should I avoid deepcopy?
        self.critic_target.load_state_dict(self.critic.state_dict())
        for k,v in self.critic_target.named_parameters():
            v.requires_grad = False#固定参数

        


class Actorv2(Actor):
    def __init__(self, *args, input_size=108, **kwargs):
        self.input_size = input_size
        super(Actorv2, self).__init__(*args,**kwargs)
        self.log_std_min = -10
        self.log_std_max = 2
        

    def get_action_dist_params(self, obs: th.Tensor, detach_encoder=False) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.
        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, detach_encoder)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        # rad中只有eval用deterministic的策略
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        #log_std = th.clamp(log_std, -10, 2)
        # here mean actions can be any real vector
        return mean_actions, log_std, {}

    def action_log_prob(self, obs: th.Tensor, detach_encoder=False) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, detach_encoder=detach_encoder)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def extract_features(self, obs: th.Tensor, detach_encoder) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs, detach_encoder)

class ContinuousCriticv2(ContinuousCritic):
    def __init__(self, *args, input_size=100, **kwargs):
        self.input_size = input_size
        super(ContinuousCriticv2, self).__init__(*args,**kwargs)

    def forward(self, obs: th.Tensor, actions: th.Tensor, detach_encoder=False) -> Tuple[th.Tensor, ...]:
        features = self.extract_features(obs, detach_encoder)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)
        
    def extract_features(self, obs: th.Tensor, detach_encoder) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs, detach_encoder)

class ReplayBufferv2(ReplayBuffer):
    def __init__(self, *args, data_aug='crop', input_size = 100, translate_mid_size = 100,\
     **kwargs):

        self.data_aug = data_aug
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.translate_mid_size = translate_mid_size
        
    def total_memory_usage(self):
        return (self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes +\
         self.dones.nbytes + self.next_observations.nbytes)/1e9

    def _get_samples(self, batch_inds: np.ndarray, env = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        
        if self.data_aug=='crop':
            obs = random_crop_vec(obs, self.input_size)
            next_obs = random_crop_vec(next_obs, self.input_size)
        elif self.data_aug=='translate':
            obs = center_crop_vec(obs, self.translate_mid_size)
            next_obs = center_crop_vec(next_obs, self.translate_mid_size)
            obs, rndm_idxs = random_translate_vec(obs, self.input_size, return_random_idxs=True)
            next_obs = random_translate_vec(next_obs, self.input_size, **rndm_idxs)
        elif self.data_aug=='no_aug':
            obs = center_crop_vec(obs, self.input_size)
            next_obs = center_crop_vec(next_obs, self.input_size)
        
        data = (
            obs,
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RAD')
    parser.add_argument('--experiment_id', type=str, metavar='string',default='test',
                        help='name of experiment')

    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--seed', default=23)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=8, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--data_aug', default='no_aug')
    parser.add_argument('--render_size', default=100, type=int)
    parser.add_argument('--learn_step', default=1000, type=int)
    parser.add_argument('--input_size', default=108, type=int, help="image size after data augmentation")

    args = parser.parse_args()

    env = make_dmc_aug_env(args, args.render_size, source=True)
    policy_kwargs = dict(
    features_extractor_class= SACEncoder,
    net_arch = dict(qf=[1024, 1024], pi=[1024, 1024]),
    share_features_extractor = True, input_size = args.input_size, channels=9)
    kwargs = dict(buffer_size=100000, learning_rate=1e-4,\
     batch_size=128, tau=0.05,ent_coef ='auto_0.1',target_update_interval=2, \
      gradient_steps=2, critic_update_steps=1, actor_update_freq = 2, \
      data_aug = args.data_aug, input_size = args.input_size, \
      translate_mid_size = 100, channels=9
     )
    print(kwargs)

    model =SACv2('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, \
    tensorboard_log='./dmc_test_sac/'+'swingup', **kwargs)
    model.learn(total_timesteps=args.learn_step)
    print(model.policy)