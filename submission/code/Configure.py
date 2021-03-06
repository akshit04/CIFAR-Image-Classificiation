# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
from Network import ImprovedBlock
model_configs = {
	"name": 'improved-model',
	"save_dir": '../saved_models/',
	"depth": 18,
	"block": ImprovedBlock,
	"num_classes": 10
	# ...
}

training_configs = {
	"learning_rate": 0.1,
	"batch_size": 128,
	"save_interval": 10,
	"weight_decay": 2e-4, 
	"max_epoch": 250,
	"lr_schedule": {100:0.1, 200:0.01, 350:0.001}
	# ...
}

### END CODE HERE
