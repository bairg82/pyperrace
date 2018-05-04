import replay_buffer

# create and display image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import environment as env

# how many data point is represented
size = 1000

# create instance
replay = replay_buffer.ReplayBuffer(1500000)

# load from saved files
replay.load()

# get a random batch
s_batch, a_batch, r_batch, t_batch, s2_batch = replay.sample_batch(size, get_all=False)

# show all
print('size:'+str(replay.count))
# size = replay.count

trk_pic = mpimg.imread('PALYA5.bmp')  # beolvassa a pályát

plt.imshow(trk_pic)
# draw someting
for i in range(size):
    plt.plot([x[2] for x in s_batch], [x[3] for x in s_batch], 'y.')

plt.draw()
plt.pause(0.1)

plt.show()
