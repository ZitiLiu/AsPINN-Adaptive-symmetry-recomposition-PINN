import Module.Training as Training
import Module.Training_3d as Training_3d

task_1 = Training.model('Burgers',1)
task_1.train()
task_1 = Training_3d.model('Thermal_3d',1)
task_1.train()