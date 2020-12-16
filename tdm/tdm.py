from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core as core
from utils.logx import EpochLogger
import torch.nn.functional as F
import math
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.goal_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.horizon = np.zeros(size, dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, goal, horizon, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.goal_buf[self.ptr] = goal
        self.horizon[self.ptr] = horizon
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     goal = self.goal_buf[idxs],
                     horizon = self.horizon[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class EpisodicReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, max_horizon):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.max_horizon = max_horizon
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim*max_horizon), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim*max_horizon), dtype=np.float32)
        self.goal_buf = np.zeros(core.combined_shape(size, act_dim*max_horizon), dtype=np.float32)
        self.horizon = np.zeros((size,max_horizon), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim*max_horizon), dtype=np.float32)
        self.rew_buf = np.zeros((size,max_horizon), dtype=np.float32)
        self.max_horizon_buf = np.zeros((size), dtype=np.float32)
        self.done_buf = np.zeros((size,max_horizon), dtype=np.float32)
        self.episode_ptr, self.ptr, self.size, self.max_size = 0, 0, 0, size

    def store(self, obs, act, rew, next_obs, goal, horizon, done):
        self.obs_buf[self.episode_ptr, self.ptr*self.obs_dim:(self.ptr+1)*self.obs_dim] = obs
        self.obs2_buf[self.episode_ptr,self.ptr*self.obs_dim:(self.ptr+1)*self.obs_dim] = next_obs
        self.goal_buf[self.episode_ptr,self.ptr*self.obs_dim:(self.ptr+1)*self.obs_dim] = goal
        self.horizon[self.episode_ptr,self.ptr] = horizon
        self.act_buf[self.episode_ptr,self.ptr*self.act_dim:(self.ptr+1)*self.act_dim] = act
        self.rew_buf[self.episode_ptr,self.ptr] = rew
        self.done_buf[self.episode_ptr,self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        

    def finish_episode(self, max_horizon):
        self.max_horizon_buf[self.episode_ptr]=max_horizon
        self.episode_ptr+=1
        self.ptr = 0
        self.size = (self.size+1) % self.max_size


    # Sample with goal relabelling
    def sample_batch(self, batch_size=32, true_goal_ratio = 0.5):

        episode_idx = np.random.randint(0, self.size, size=batch_size)
        random_floats = (np.random.uniform(size=batch_size).reshape(-1) * self.max_horizon_buf[episode_idx])
        state_idx = random_floats.astype(int)
        future_goals = (np.random.uniform(size=batch_size).reshape(-1) * (self.max_horizon_buf[episode_idx]-state_idx)+state_idx).astype(int)
        future_goals = np.maximum(future_goals,self.max_horizon_buf[episode_idx]-1).astype(int)
        rand_horizons = np.random.randint(1,self.max_horizon,batch_size)
        true_goal_binary = np.random.uniform(size=batch_size)>true_goal_ratio

        obs = np.zeros((batch_size,self.obs_dim))
        obs2 = np.zeros((batch_size,self.obs_dim))
        goal = np.zeros((batch_size,self.obs_dim))
        horizon = np.zeros((batch_size))
        act = np.zeros((batch_size,self.act_dim))
        rew = np.zeros((batch_size))
        done = np.zeros((batch_size))
        for i in range(batch_size):
            obs[i,:] = self.obs_buf[episode_idx[i],state_idx[i]*self.obs_dim:(state_idx[i]+1)*self.obs_dim]
            obs2[i,:] = self.obs2_buf[episode_idx[i],state_idx[i]*self.obs_dim:(state_idx[i]+1)*self.obs_dim]
            act[i,:] = self.act_buf[episode_idx[i],state_idx[i]*self.act_dim:(state_idx[i]+1)*self.act_dim]
            goal[i,:] = true_goal_binary[i] * self.goal_buf[episode_idx[i],state_idx[i]*self.obs_dim:(state_idx[i]+1)*self.obs_dim]+\
                        (1-true_goal_binary[i])*self.obs_buf[episode_idx[i],future_goals[i]*self.obs_dim:(future_goals[i]+1)*self.obs_dim]
            
            horizon[i] = true_goal_binary[i] * self.horizon[episode_idx[i],state_idx[i]] + (1-true_goal_binary[i])*rand_horizons[i]
            rew = self.rew_buf[episode_idx[i],state_idx[i]]
            done = self.done_buf[episode_idx[i],state_idx[i]]

        batch = dict(obs=obs,
                     obs2=obs2,
                     goal = goal,
                     horizon = horizon,
                     act=act,
                     rew=rew,
                     done=done)



        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
class TDM:

    def __init__(self, env_fn, actor_critic=core.MLPtdmActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=2000, epochs=1000, replay_size=int(100000), gamma=0.99, 
        polyak=0.995, lr=1e-3, p_lr=1e-3, alpha=0.0, batch_size=100, start_steps=1000, 
        update_after=1000, update_every=50, num_test_episodes=20, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, algo='SAC'):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

            """

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space,special_policy='tdm', **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.gamma  = gamma


        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.max_horizon = 30
        # Experience buffer
        self.replay_buffer = EpisodicReplayBuffer(obs_dim=self.obs_dim[0], act_dim=self.act_dim, size=replay_size,max_horizon=self.max_horizon)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        self.algo = algo
        self.start_steps = start_steps


        self.alpha = alpha # CWR does not require entropy in Q evaluation
        self.target_update_freq = 1
        self.p_lr = 1e-3
        self.lr = 1e-3

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = self.env._max_episode_length
        self.epochs= epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak
        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        
        self.eval_freq= 100



    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        g,h = data['goal'],data['horizon']
        q1 = self.ac.q1(torch.cat([o,g,h.view(-1,1)],axis=1),a)
        q2 = self.ac.q2(torch.cat([o,g,h.view(-1,1)],axis=1),a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(torch.cat([o2,g,(h-1).view(-1,1)],axis=1))

            # Target Q-values
            q1_pi_targ = torch.abs(self.ac_targ.q1(torch.cat([o2,g,(h-1).view(-1,1)],axis=1), a2)-g)
            q2_pi_targ = torch.abs(self.ac_targ.q2(torch.cat([o2,g,(h-1).view(-1,1)],axis=1), a2)-g)
            q_pi_targ = torch.max(q1_pi_targ,q2_pi_targ)
            # q_pi_targ = torch.max(q1_pi_targ, q2_pi_targ)
            dist_to_goal = torch.min(torch.abs(o2 - g), torch.abs(2*math.pi - torch.max(o2,g) + torch.min(o2,g)))
            backup = ((h-1)==0).view(-1,1)*dist_to_goal + ((h-1)!=0).view(-1,1)*q_pi_targ
            # backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((torch.abs(q1-g.detach()) - backup)**2).mean()
        loss_q2 = ((torch.abs(q2-g.detach()) - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        g = data['goal']
        h = data['horizon']
        # Sample recent data for policy update
        pi, logp_pi = self.ac.pi(torch.cat([o,g,h.view(-1,1)],axis=1))
        # q1_pi = self.ac.q1(torch.cat([o,g,h.view(-1,1)],axis=1), pi)
        # q2_pi = self.ac.q2(torch.cat([o,g,h.view(-1,1)],axis=1), pi)
        # q_pi = torch.min(q1_pi, q2_pi)

        q_pi =  - torch.abs(self.ac.q1(torch.cat([o,g,h.view(-1,1)],axis=1), pi)-g.detach())

        loss_pi = (self.alpha * logp_pi.view(-1,1) - q_pi.sum(1)).mean()
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info



    def update(self,data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False




        # data = self.replay_buffer.sample_batch(self.batch_size)
        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        if update_timestep%self.target_update_freq==0:
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def get_q_value(self,o,a):
        return self.ac.q1(torch.as_tensor(o, dtype=torch.float32), 
                      torch.as_tensor(a, dtype=torch.float32))

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def test_tdm(self):
        goal_reaches = 0
        for j in range(self.num_test_episodes):
            goal = self.test_env.sample_random_goal()    
            horizon = np.random.randint(1,self.max_horizon)
            q_values = []
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            goal_reached = False
            for t in range(horizon,0,-1):   
                a = self.get_action(np.concatenate((o,goal,np.array([t]))),True)
                q_val = self.get_q_value(np.concatenate((o,goal,np.array([t]))).reshape(1,-1), a.reshape(1,-1)).detach().cpu().numpy()
                q_values.append(q_val)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                if((o-goal).sum()<(0.1*self.obs_dim[0])):
                    goal_reached=True
            if(goal_reached):
                goal_reaches+=1

            q_errors = np.abs(np.abs(q_values-goal)-np.abs(o-goal))
            # q_errors = np.abs(np.array(q_values).squeeze()+np.abs(o-goal).reshape(1,-1))
            self.logger.store(TDMError=np.sum(q_errors),TestEpRet=ep_ret)
        self.logger.store(GoalReach=goal_reaches/self.num_test_episodes) 


    def run(self):
        total_steps = self.steps_per_epoch * self.epochs
        total_episodes = 100000
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        timesteps = 0
        for e in range(total_episodes):
            # Sample a goal
            o = self.env.reset()
            goal = self.env.sample_random_goal()
            # Sample a horizon
            horizon = np.random.randint(1,self.max_horizon)
            # print("Training episode: {}".format(e))
            obs_list = []
            for t in range(horizon,0,-1):
                # Until start_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration. Afterwards, 
                # use the learned policy. 
                if timesteps > self.start_steps:
                    a = self.get_action(np.concatenate((o,goal,np.array([t]))))
                else:
                    a = self.env.action_space.sample()                

                # Step the env
                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                timesteps+=1
            

                # Store experience to replay buffer
                self.replay_buffer.store(o, a, r, o2, goal, t, d)
                obs_list.append(o)
                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2

                # # End of trajectory handling
                # if d or (ep_len == self.max_ep_len):
                #     o, ep_ret, ep_len = self.env.reset(), 0, 0

                # Update handling
                if timesteps >= self.update_after and timesteps % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        self.update(data=batch,update_timestep=t)

                # End of epoch handling
                if (timesteps+1) % self.steps_per_epoch == 0:
                    epoch = timesteps//self.steps_per_epoch

                    # Save model
                    
                    self.logger.save_state({'env': self.env}, None)

                    # Test the performance of the deterministic version of the agent.
                    self.test_tdm()
                    # Log info about epoch
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('Episodes', e)
                    self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                    self.logger.log_tabular('TDMError', with_min_and_max=True)
                    self.logger.log_tabular('GoalReach', average_only=True)
                    self.logger.log_tabular('TotalUpdates', timesteps)
                    self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                    self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                    self.logger.log_tabular('LogPi', with_min_and_max=True)
                    self.logger.log_tabular('LossPi', average_only=True)
                    self.logger.log_tabular('LossQ', average_only=True)
                    self.logger.log_tabular('Time', time.time()-start_time)
                    self.logger.dump_tabular()


            self.replay_buffer.finish_episode(horizon)

            if(e%1000==0):
                print(obs_list)
                print("Goal is: {}".format(goal))
