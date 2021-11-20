### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
# from matplotlib import pyplot as plt
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
import csv

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("--result_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	print(args.mode, args.data_dir)
	model = MyModel(model_configs)
	# model.load()
	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
		
		train_stats = model.train(x_train, y_train, training_configs, x_valid, y_valid)
		w = csv.writer(open(os.path.join(model_configs["save_dir"], model_configs['name'])+".csv", "w"))
		for key, val in train_stats.items():
			w.writerow([key, val])

		# SINGLE
		# score,loss = model.evaluate(x_test, y_test)
		# print("The test score is: {:.3f}% ({:.4f})".format(score*100, loss))

		# MULTIPLE
		checkpoint_num_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
		for checkpoint_num in checkpoint_num_list:
			model.load(checkpoint_num)
			score,loss = model.evaluate(x_test, y_test)
			print("The test score for Checkpoint {:d} is: {:.3f}% ({:.4f})".format(checkpoint_num, score*100, loss))
		
		# score,loss = model.evaluate(x_test, y_test)
		# print("The test score is: {:.3f}% ({:.4f})".format(score*100, loss))

	elif args.mode == 'test':
		best_checkpoint = 160
		model.load(best_checkpoint)
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)

		# model.evaluate(x_test, y_test)
		score,loss = model.evaluate(x_test, y_test)
		print("The test score is: {:.3f}% ({:.4f})".format(score*100, loss))

	elif args.mode == 'predict':
		best_checkpoint = 160
		model.load(best_checkpoint)

		# Predicting and storing results on private testing dataset 
		x_test = load_testing_images(args.data_dir)
		predictions = model.predict_prob(x_test)
		np.save(args.result_dir, predictions)
		

### END CODE HERE

