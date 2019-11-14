from __future__ import print_function
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices

import random                # Handling random number generation
import time
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys

tf.compat.v1.disable_eager_execution()
sys.path.append("game/")

from Memory import Memory
from SumTree import SumTree
import wrapped_flappy_bird as gamer
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

'''
1. Convert image to grayscale
2. Resize image to 80x80
3. Stack last 4 frames to produce an 80x80x4 input array for network
'''

def preprocess_frame(frame):
    # Crop the screen (remove part that contains no information)
    # [Up: Down, Left: right]
    frame = skimage.color.rgb2gray(frame)
    frame = skimage.transform.resize(frame,(80,80))
    frame = skimage.exposure.rescale_intensity(frame,out_range=(0,255))
    frame = frame / 255.0

    return frame # 80x80x1 frame

stack_size = 4 # We stack 4 frames
stacked_frames  =  deque([np.zeros((80,80), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    
    if is_new_episode:

        stacked_frames = deque([np.zeros((80,80), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

#MODEL HYPERPARAMETERS
state_size = [80,80,4] #(width, height, channels)
action_size = 2 #2 possible actions
learning_rate = 0.00025

#TRAINING HYPERPARAMETERS
total_episodes = 5000
max_steps = 5000
batch_size = 64

#FIXED Q TARGET HYPERPARAMTERS
max_tau = 10000

#Exploration parameters for epsilon greedy strategy
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005 #exponential decay rate for exploration prob

#Q Learning hyperparameters
gamma = 0.95 #discounting rate

#MEMORY HAPERPARAMETERS
pretrain_length = 100000#number of expeiences stoed in the Memory when initialized for the first time
memory_size = 100000

#ADJUSTABLE
training = False
episode_render = False

FRAME_PER_ACTION = 1

class DDDQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.compat.v1.variable_scope(self.name):

            #create placeholder
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs_")
            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name="ISWeights_")
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, action_size], name="actions_")

            #target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name='target')


            #Input is 80*80*4
            self.conv1 = tf.compat.v1.layers.conv2d(inputs = self.inputs_,
                                        filters = 32,
                                        kernel_size = [8,8],
                                        strides = [4,4],
                                        padding = "VALID",
                                        kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                        name = "conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            ## --> [20, 20, 32]

            self.conv2 = tf.compat.v1.layers.conv2d(inputs = self.conv1_out,
                                        filters = 64,
                                        kernel_size = [4,4],
                                        strides = [2,2],
                                        padding = "VALID",
                                        kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                        name = "conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.compat.v1.layers.conv2d(inputs = self.conv2_out,
                                        filters = 64,
                                        kernel_size = [3,3],
                                        strides = [1,1],
                                        padding = "VALID",
                                        kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                        name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.compat.v1.layers.flatten(self.conv3_out)
            ## Here we separate into two streams
            #The one that calculate V(s)

            self.value_fc = tf.compat.v1.layers.dense(inputs = self.flatten,
                                    units = 256,
                                    activation = tf.nn.elu,
                                    kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    name = "value_fc")

            self.value = tf.compat.v1.layers.dense(inputs = self.value_fc,
                                    kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    units = 1,
                                    activation = None,
                                    name = "value")

            #The one that calculate A(s,a)
            self.advantage_fc = tf.compat.v1.layers.dense(inputs = self.flatten,
                                    units = 256,
                                    activation = tf.nn.elu,
                                    kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    name = "advantage_fc")

            self.advantage = tf.compat.v1.layers.dense(inputs = self.advantage_fc,
                                    kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                    units = self.action_size,
                                    activation = None,
                                    name = "advantage")

            #Agregating layer
            #Q(s,a) = V(s) +(A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(input_tensor=self.advantage, axis=1, keepdims=True))

            #Q is our predicted Q value
            self.Q = tf.reduce_sum(input_tensor=tf.multiply(self.output, self.actions_), axis=1)

            #The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)

            self.loss = tf.reduce_mean(input_tensor=self.ISWeights_ * tf.math.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.compat.v1.reset_default_graph()

#Instantiate the DQNetwork
DQNetwork = DDDQNetwork(state_size, action_size, learning_rate, name="DQNetwork")

#Instantiate the target network
TargetNetwork = DDDQNetwork(state_size, action_size, learning_rate, name="TargetNetwork")

#deal with the empty memory problem bu pre-populating our emory by taking random actions
memory = Memory(memory_size)
# Render the environment
game_state = gamer.GameState()
nothing = [1, 0]
up = [0, 1]
possible_actions = [nothing, up]

if training == True:
	for i in range(pretrain_length):
	    # If it's the first step
	    if i == 0:        # First we need a state
	        do_nothing = [1, 0]
	        state, reward, done = game_state.frame_step(do_nothing)
	        state, stacked_frames = stack_frames(stacked_frames, state, True)
	    
	    # Random action
	    index = random.randrange(10);
	    if (index<9):
	    	action = [1,0]
	    else:
	    	action = [0,1]

	    # Get the rewards
	    next_state, reward, done = game_state.frame_step(action)
	    
	    # If we're dead
	    if done:
	        # We finished the episode
	        next_state = np.zeros(state.shape)
	        
	        # Add experience to memory
	        #experience = np.hstack((state, [action, reward], next_state, done))
	        experience = state, action, reward, next_state, done
	        memory.store(experience)
	        
	        # Start a new episode
	        game_state = gamer.GameState()

	        # First we need a state
	        do_nothing = [1, 0]
	        state, reward, done = game_state.frame_step(do_nothing)
	        state, stacked_frames = stack_frames(stacked_frames, state, True)
	        
	    else:
	        # Get the next state
	        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
	        
	        # Add experience to memory
	        experience = state, action, reward, next_state, done
	        memory.store(experience)
	        
	        # Our state is now the next_state
	        state = next_state

#Setup TensorBoard Writer
writer = tf.compat.v1.summary.FileWriter("/tensorboard/dddqn/1")

##Losses
tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.compat.v1.summary.merge_all()

'''
Train the Agent
'''
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    #EPSILON GREEDY STRATEGY
    exp_exp_tradeoff = np.random.rand()

    #improved EG
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate*decay_step)

    if (explore_probability > exp_exp_tradeoff):
        #print("----------Random Action----------")
	    index = random.randrange(10);
	    if (index<9):
	    	action = [1,0]
	    else:
	    	action = [0,1]
        #print(action)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability

#This function helps us to copy one set of variables to another
def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    
    # Get the parameters of our Target_network
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder




# Saver will help us to save our model
saver = tf.compat.v1.train.Saver()

if training == True:
    with tf.compat.v1.Session() as sess:
        # Initialize the variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        decay_step = 0
        tau = 0

        
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        
        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            
            # Make a new episode and observe the first state
            game_state = gamer.GameState()
            do_nothing = [1, 0]
            state, reward, done = game_state.frame_step(do_nothing)

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
            while step < max_steps:
                step += 1
                tau += 1
                decay_step +=1
                
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # Do the action
                next_state, reward, done = game_state.frame_step(action)
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((120,140), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                
                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                
                # Get Q values for next_state 
                q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb})
                
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    
                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
              
                
                
                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                
                
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                
                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save model every 100 episodes
            if episode % 100 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
            
'''
Step 9: Watch our Agent play
'''

if training == False:
    with tf.compat.v1.Session() as sess:
        
        
        # Load the model
        saver.restore(sess, "./models/model.ckpt")
        
        for i in range(10):
            
            total_reward = 0
            game_state = gamer.GameState()

            do_nothing = [1, 0]

            state, reward, done = game_state.frame_step(do_nothing)
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            total_reward += reward

            while not done:
                ## EPSILON GREEDY STRATEGY
                # Choose action a from state s using epsilon greedy.
                ## First we randomize a number
                exp_exp_tradeoff = np.random.rand()
                

                explore_probability = 0.0001
        
                if (explore_probability > exp_exp_tradeoff):
                    # Make a random action (exploration)
                    action = random.choice(possible_actions)
            
                else:
                    # Get action from Q-network (exploitation)
                    # Estimate the Qs values state
                    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
            
                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)
                    action = possible_actions[int(choice)]
                
                next_state, reward, done = game_state.frame_step(action)
                total_reward += reward

                if done:
                    break
                    
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    state = next_state
            
            print("Score: ", total_reward)
        
       

