#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np 
from model import twoStreamNet_VGG16
from data import Dataset

train_params = {
	'num_classes' : 51,
	'batch_size' : 16,
	'n_epochs': 40,
	
}
