import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Helper libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def s(x):
    
    "Activation function to implement the basis functions, cubic"
    
    y0 = K.switch(tf.logical_and(tf.zeros(tf.shape(x))<=x, x<tf.ones(tf.shape(x))), 
                 x**3/6, 
                 tf.zeros(tf.shape(x))
                )
    
    y1 = K.switch(tf.logical_and(tf.ones(tf.shape(x))<=x, x<2*tf.ones(tf.shape(x))),
                 (-3.*(x-1.)**3 +3.*(x-1.)**2 + 3*(x-1.)+1.)/6.,
                 tf.zeros(tf.shape(x))
                )
    
    y2 = K.switch(tf.logical_and(tf.ones(tf.shape(x))*2<=x, x<3*tf.ones(tf.shape(x))),
                 (3*(x-2)**3 - 6*(x-2)**2 + 4. )/6.,
                 tf.zeros(tf.shape(x))
                )

    y3 = K.switch(tf.logical_and(tf.ones(tf.shape(x))*3<=x, x<4*tf.ones(tf.shape(x))),
                 ( 4. -x)**3/6.,
                 tf.zeros(tf.shape(x))
                )    
    
    y  = y0 + y1 + y2 + y3
    
    return y

def partition_weights(n,n0):
    p0 = n0*n
    y = np.zeros((n,p0))
    for i in range(n):
        y[i,i*n0:(i+1)*n0] = 1.
    return y 

def partition_bias(n,n0):
    p0 = n0*n
    return (np.arange(0.,p0)%n0)    

def spline_function_(n,m,c0,u0):
    """Cubic spline"""
    
    d0 = 3 # if the degree is 3 then use the activation function s3(), if it's 2 then use s()
 
    n0 = c0 + d0
    p0 = n0*n

    inn = tf.keras.layers.Input(shape=(n,))

    l00 = tf.keras.layers.Dense(activation=s,
                                units=p0,
                                use_bias=True,
                                trainable=False,
                                kernel_initializer=tf.constant_initializer(c0*partition_weights(n,n0)),
                                bias_initializer=tf.constant_initializer(d0 - partition_bias(n,n0)) 
                               )(inn)
    
    out = tf.keras.layers.Dense(units=m,
                                use_bias=False,
                                trainable=True,
                                kernel_initializer=tf.constant_initializer(u0)
                               )(l00)

    model = tf.keras.Model(inputs=inn, 
                           outputs= out
                          )
    return model


print(tf.__version__)

x = np.arange(0.,4,0.001)
y = [s(x0) for x0 in x]

df_activation = pd.DataFrame(dict(x=x, S=np.array(y)))
#df_activation

fig = plt.figure(1)

sns.lineplot(x='x',y='S',data=df_activation,
             color=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.65),
             linewidth=5)
fig.gca().set(xlabel=r'x', ylabel=r'Uniform Cubic B-Spline S(x)')
fig.savefig("KASAM_Theory/uniform_cubic_b_spline_activation_function.png", close = True, verbose = True)

plt.close()

r = 0.6
x0 = np.arange(0.-r,1.+r, 0.001)
my_dict = dict(x=x0)

for j in range(0,8):
    #
    # Initialise a model
    model = spline_function_(1,1,5,0.)

    # Adjust weights for testing purposes, comment out otherwise
    weights = model.get_weights()
    #print(weights)
    weights[2][j] = 1.
    model.set_weights(weights)

    my_dict[str(j)] = model.predict(x0).flatten()
    
df_uniform_spline_basis = pd.DataFrame(my_dict)
#df_uniform_spline_basis

#fig = plt.figure(10)
fig = plt.figure(10)
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
for j in range(0,8):
    sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                 color=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.075*j+0.2),linewidth=5)
    #ns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
fig.gca().set(xlabel=r'x', ylabel=r'$S_{i}(x)$')
fig.savefig("KASAM_Theory/uniform_cubic_b_spline_basis_functions.png", 
            close = True, verbose = True, dpi=500,bbox_inches='tight')
plt.close()

index0 = 3

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

df_active_point = pd.DataFrame(dict(x=np.array([0.3]), y=np.array([0.])))

fig = plt.figure(10)

g = sns.scatterplot(data=df_active_point, x="x", y="y",color=sns.color_palette("tab10")[1],
                    zorder=20,label="Input Point",s=120)
g.set_xticks(np.arange(-0.5,2.0,0.5))
g.set_yticks(np.arange(0.,1.7,0.1))

#sns.lineplot(x='x',y='y',data=df_true_function,color=sns.color_palette('Greens')[index0],linewidth=2)

for j in range(0,8):
    if j == 0:
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.25)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=2,label="Inactive Basis Function")
    elif j == 1:
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.75)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=4,zorder=10,label="Active Basis Function")
    elif ((j>1) and (j<=4)):
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.75)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=4,zorder=10)
    else:
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.25)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=2)
    
legend = plt.legend(loc='upper right')
frame = legend.get_frame()
frame.set_facecolor('white')
plt.ylim(-0.05,0.95)
plt.xlim(-0.6,1.6)

fig.gca().set(xlabel=r'x', ylabel=r'$S_{i}(x)$')
fig.savefig("KASAM_Theory/SAM_properties_1_2_Proof.png", close = True, verbose = True, dpi=500,bbox_inches='tight')

plt.close()

r = 0.6
x0 = np.arange(0.-r,1.+r, 0.001)
my_dict = dict(x=x0)

number_basis_functions = 16 # must be more than 4

for j in range(0,number_basis_functions):
    #
    # Initialise a model
    model = spline_function_(1,1,number_basis_functions-3,0.)

    # Adjust weights for testing purposes, comment out otherwise
    weights = model.get_weights()
    #print(weights)
    weights[2][j] = 1.
    model.set_weights(weights)

    my_dict[str(j)] = model.predict(x0).flatten()
    
df_uniform_spline_basis = pd.DataFrame(my_dict)
#df_uniform_spline_basis

index0 = 3

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

df_active_point = pd.DataFrame(dict(x=np.array([0.11]), y=np.array([0.])))
df_active_point2 = pd.DataFrame(dict(x=np.array([.96]), y=np.array([0.])))

fig = plt.figure(10)

g = sns.scatterplot(data=df_active_point, x="x", y="y",color=sns.color_palette("tab10")[1],
                    zorder=20,label="Input Points",s=120)
sns.scatterplot(data=df_active_point2, x="x", y="y",color=sns.color_palette("tab10")[1],
                    zorder=20,s=120)
g.set_xticks(np.arange(-0.5,2.0,0.15))
g.set_yticks(np.arange(0.,1.7,0.1))

#sns.lineplot(x='x',y='y',data=df_true_function,color=sns.color_palette('Greens')[index0],linewidth=2)

for j in range(0,16):
    if j == 0:
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.25)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=2,label="Inactive Basis Function")
    elif j == 1:
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.75)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=4,zorder=10,label="Active Basis Function")
    elif ((j>1) and (j<=4)):
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.75)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=4,zorder=10)
    elif (j>11):
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.75)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=4,zorder=10)        
    else:
        colorchoice = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.25)
        sns.lineplot(x='x',y=str(j),data=df_uniform_spline_basis,\
                     color=colorchoice,linewidth=2)
    
legend = plt.legend(loc='upper right')
frame = legend.get_frame()
frame.set_facecolor('white')
plt.ylim(-0.05,0.95)
plt.xlim(-0.25,1.25)

fig.gca().set(xlabel=r'x', ylabel=r'$S_{i}(x)$')
fig.savefig("KASAM_Theory/SAM_properties_3_Proof.png", close = True, verbose = True, dpi=500,bbox_inches='tight')

plt.close()

rs = np.random.RandomState(16989)
x = rs.rand(10)
y0 = np.sin(2.*np.pi*x)

dx = 0.001
index0 = 3
x1 = np.arange(0.,1.+dx,dx)
y1 = np.sin(2.*np.pi*x1)

df_training_set = pd.DataFrame(dict(x=x, y=y0))
df_true_function = pd.DataFrame(dict(x=x1, y=y1))

dict_strat = dict()
dict_strat['x'] = x1

#strat_number = [1,5,13,29,61,125,253,509]
strat_number = [5,13,253]

for strat_example in strat_number:
    # [1,5,13,29,61,125,253,509]
    model = spline_function_(1,1,strat_example,0.)


    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.075),
                  loss=tf.keras.losses.mean_squared_error,metrics=['mse'])
    model.fit(x, y0, epochs=5,verbose=0)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05),
                  loss=tf.keras.losses.mean_squared_error,metrics=['mse'])
    model.fit(x, y0, epochs=10,verbose=0)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.mean_squared_error,metrics=['mse'])
    model.fit(x, y0, epochs=40,verbose=0)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.mean_squared_error,metrics=['mse'])
    model.fit(x, y0, epochs=80,verbose=0)

    dict_strat[str(strat_example)] = model.predict(x1).flatten()

dx = 0.001
index0 = 3
x1 = np.arange(0.,1.+dx,dx)
y1 = np.sin(2.*np.pi*x1)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

df_training_set = pd.DataFrame(dict(x=x, y=y0))
df_true_function = pd.DataFrame(dict(x=x1, y=y1))
df_stratification = pd.DataFrame(dict_strat)

fig = plt.figure(10)

g = sns.scatterplot(data=df_training_set, x="x", y="y",color=sns.color_palette("tab10")[1],
                    zorder=10,label="Data",s=100)
g.set_xticks(np.arange(0.,1.5,0.5))
g.set_yticks(np.arange(-1.,1.5,1.))

#sns.lineplot(x='x',y='y',data=df_true_function,color=sns.color_palette('Greens')[index0],linewidth=2)

for j in strat_number:
    sns.lineplot(x='x',y=str(j),data=df_stratification,\
                 color=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)(0.3*strat_number.index(j)+0.),
                 linewidth=2.5,zorder=strat_number.index(j)+5,label="K = "+str(j+3))

    
legend = plt.legend()
frame = legend.get_frame()
frame.set_facecolor('white')

fig.gca().set(xlabel=r'$x$', ylabel=r'$y$')
fig.savefig("KASAM_Theory/uniform_cubic_b_spline_stratification.png", 
            close = True, verbose = True, dpi=500,bbox_inches='tight')

plt.close()

k = 8

xv = np.linspace(0.1,0.9,2*k)

r0 = np.random.RandomState(1989)
g0 = np.random.RandomState(6989)
x0 = xv[:k]
y0 = 2.*x0 + g0.randn(k)*0.15

r1 = np.random.RandomState(1189)
g1 = np.random.RandomState(1729)
x1 = xv[k:]
y1 = 0. + g1.randn(k)*0.1 +  0.93

index0 = 3

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

df_active_point00 = pd.DataFrame(dict(x=x0, y=y0))

fig = plt.figure(10)

g = sns.scatterplot(data=df_active_point00, x="x", y="y",color=sns.color_palette("tab10")[0],
                    zorder=20,label="First Task",s=160)
g.set_xticks(np.arange(-0.5,2.0,0.25))
g.set_yticks(np.arange(0.,1.7,0.1))


df_true_function_00 = pd.DataFrame(dict(x=np.array([-2,2]),y=np.array([-4,4-0.07])))

sns.lineplot(x='x',y='y',data=df_true_function_00,color=sns.color_palette("tab10")[0],linewidth=3)

df_active_point01 = pd.DataFrame(dict(x=x1, y=y1))

fig = plt.figure(10)
#fig = plt.figure(figsize=(10,10))

g = sns.scatterplot(data=df_active_point01, x="x", y="y",color=sns.color_palette("tab10")[1],
                    zorder=20,label="Second Task",s=160)
g.set_xticks(np.arange(0.,1.1,0.25))
g.set_yticks(np.arange(0.,1.3,0.25))

df_true_function_01 = pd.DataFrame(dict(x=np.array([-2,2]),y=np.array([0.95,0.95])))

sns.lineplot(x='x',y='y',data=df_true_function_01,color=sns.color_palette("tab10")[1],linewidth=3)
    
legend = plt.legend(loc='lower right')
frame = legend.get_frame()
frame.set_facecolor('white')
plt.ylim(-0.0,1.25)
plt.xlim(0.,1.)

fig.gca().set(xlabel=r'x', ylabel=r'$y(x)$')
fig.savefig("KASAM_Theory/Linear_functions_catastrophic_interference.png", 
            close = True, verbose = True, dpi=500,bbox_inches='tight')

plt.close()

dx = 0.001
index0 = 3
x1 = np.arange(0.,1.+dx,dx)
y1 = np.sin(2.*np.pi*x1)

# Initialise a model
model = spline_function_(1,1,13,0.)

output1 = model.predict(x1).flatten()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),loss=tf.keras.losses.mean_squared_error,metrics=['MSE'])

rt = np.random.RandomState(1789)
x2 = rt.rand(1000)
y2 = np.sin(2.*np.pi*x2)

model.fit(x2, y2, epochs=10,verbose=0)

output2 = model.predict(x1)

rs1 = np.random.RandomState(1189)
x3 = rs1.rand(30)
y3 = model.predict(x3)

rs2 = np.random.RandomState(1149)
x4 = 0.5*rs2.rand(30)+0.5
y4 = 0*x4

probability_new_data = 0.5
num_data = 5000

train_inputs_new_data = x4.copy()
train_labels_new_data = y4.copy()

index_choice =  np.random.randint(0,len(train_labels_new_data),num_data)

new_input_samples = train_inputs_new_data[index_choice]
new_label_samples = train_labels_new_data[index_choice]


train_inputs_mem = x3.copy()
train_labels_mem = y3.flatten()

index_choice2 =  np.random.randint(0,len(train_inputs_mem),num_data)
mem_input_samples = train_inputs_mem[index_choice2]
mem_label_samples = train_labels_mem[index_choice2]

reveries_input = np.zeros(num_data).astype('float32')
reveries_label = np.zeros(num_data).astype('float32')

reverie_constructor = np.random.choice([True,False],
                                num_data,
                                p=[probability_new_data,1.-probability_new_data]
                               )

reveries_input[reverie_constructor] = new_input_samples[reverie_constructor]
reveries_label[reverie_constructor] = new_label_samples[reverie_constructor]

reveries_input[np.logical_not(reverie_constructor)] = mem_input_samples[np.logical_not(reverie_constructor)]
reveries_label[np.logical_not(reverie_constructor)] = mem_label_samples[np.logical_not(reverie_constructor)]

model.fit(reveries_input, reveries_label, epochs=20,verbose=0)

output3 = model.predict(x1)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

df_initial_model = pd.DataFrame(dict(x=x1, y=output2.flatten()))
df_long_mem = pd.DataFrame(dict(x=x3, y=y3.flatten()))
df_shrt_mem = pd.DataFrame(dict(x=x4, y=y4))
df_reveries = pd.DataFrame(dict(x=np.concatenate((x3,x4)),y=np.concatenate((y3.flatten(),y4))))
df_final_model = pd.DataFrame(dict(x=x1, y=output3.flatten()))

fig = plt.figure(20)

g = sns.scatterplot(data=df_long_mem, x="x", y="y",
                    color=sns.color_palette("tab10")[0],
                    zorder=10,label="Initial Memory",s=80)
g.set_xticks(np.arange(0.,1.5,0.5))
g.set_yticks(np.arange(-1.,1.5,1.))

sns.scatterplot(data=df_shrt_mem, x="x", y="y",
                    color=sns.color_palette("tab10")[1],
                    zorder=10,label="New Target Values",s=80)

sns.scatterplot(data=df_reveries, x="x", y="y",color='black',zorder=60,
                label="Augmented Data",s=10)

m = sns.lineplot(x='x',y='y',data=df_initial_model,\
                 color=sns.color_palette("tab10")[0],
                 linewidth=1.5,zorder=5,label="Initial Model")
#sns.lineplot(x='x',y='y',data=df_true_function,color=sns.color_palette('Greens')[index0],linewidth=2)

sns.lineplot(x='x',y='y',data=df_final_model,\
                 color='black',
                 linewidth=1.5,zorder=50,label="Updated Model")
    
legend = plt.legend()
frame = legend.get_frame()
frame.set_facecolor('white')

fig.gca().set(xlabel=r'$x$', ylabel=r'$y$')
fig.savefig("KASAM_Theory/pseudo_rehearsal.png", close = True, verbose = True, dpi=500,bbox_inches='tight')

plt.close()
