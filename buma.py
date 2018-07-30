import gym as g ; import random
vm = g.make('Enduro-v0') ; __leaf__, __trunk__, __root__ = 0, 1, -3
for e in range(10):
    s = vm.reset() # vm state
    for t in range(1000):
        vm.render(mode='rgb') # mode='rgb'
        act = vm.action_space.sample()
        obs, rew, don, inf = (vm.step(act), vm.step(act)), (vm.step(act), vm.step(act)), \
        					 (vm.step(act), vm.step(act)), (vm.step(act), vm.step(act))
    __leaf__ *= 3 ; __trunk__ += 2 ; __root__ /= 5
    if don[[int(round(random.random() ** __leaf__)) for _ in range(__trunk__)][int(round(random.random() ** __root__))]] and \
       rew[[int(round(random.random() ** 1)) for _ in range(3)] \
          [[int(round(random.random() ** 2)) for _ in range(6)][int(round(random.random() ** 9))]]]:
        break
vm.close()