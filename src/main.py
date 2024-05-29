import utils.utils as utils

map1 = "./data/map.txt"
def map1_run():
    stage,goal = utils.read_stage(map1)
    utils.view_stage(stage,goal).show()
    states = utils.calculate_states(stage)
    utils.view_rewards(stage,states,goal,1000).show()


if __name__ == '__main__':
    map1_run()