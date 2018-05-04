import replay_buffer

# create and display image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import environment as env1
import numpy as np


fig, ax = plt.subplots()
ax.plot(np.random.rand(10))
def convert(x):
    return int(x)

print(convert(1.02))
def onclick(event):
    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
     #     ('double' if event.dblclick else 'single', event.button,
     #      event.x, event.y, event.xdata, event.ydata))
    x = convert(event.xdata)
    y = convert(event.ydata)

    pos = [x, y]
    ont, int, outt = env.is_on_track(pos)
    if ont:
        try:
            dist_in, pos_in, dist_out, pos_out = env.get_pos_ref_on_side(pos)
            # print(str(dist_in) + ' ' + str(pos_in) + ' ' + str(dist_out) + ' ' + str(pos_out))
            print(str(convert(dist_in/3.835)) + ' ' + str(convert(dist_out/4.373)))
        except:
            pass

cid = fig.canvas.mpl_connect('button_press_event', onclick)

env = env1.PaperRaceEnv('h1', ref_calc='default', car_name='Touring', random_init=False)
env.reset()
env.start_game()

plt.draw()
plt.pause(0.1)

plt.show()
