import matplotlib.pyplot as plt
from utils.collector import data_collector


if __name__ == '__main__':
    data_record = data_collector(0)
    plot_list = ['pos', 'vel', 'att', 'pqr', 'torque', 'throttle', 'obs_in', 'obs_out']
    '''
    One can choose curves that you want to display.
    For example:
        If one sets plot_list = ['pos', 'vel'], then only positions (ref and real) and velocities (ref and real) are plotted.
        If one sets plot_list as shown above, then all curves (ref and real) are plotted
    '''

    dir_name = './2023-09-21-21-13-18/'     # select the directory
    data_record.load_file(dir_name)         # load file into data collector

    data_record.plot_att()                  # plot curves as in quad_att_ctrl.py
    data_record.plot_torque()
    data_record.plot_inner_obs()
    plt.show()
