# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:52:39 2014

@author: inbar
"""

import socket
from enum import Enum

class BlitzerDatasetDomain(Enum):
    apparel = 'apparel'
    automotive = 'automotive'
    baby = 'baby'
    beauty = 'beauty'
    books = 'books'
    camera = 'camera_&_photo'
    cellphones = 'cell_phones_&_service'
    videogames = 'computer_&_video_games'
    dvd = 'dvd'
    electronics = 'electronics'
    gourmet = 'gourmet_food'
    grocery = 'grocery'
    health = 'health_&_personal_care'
    jewelry = 'jewelry_&_watches'
    kitchen = 'kitchen_&_housewares'
    magazines = 'magazines'
    music = 'music'
    musicalInstruments = 'musical_instruments'
    office = 'office_products'
    outdoor = 'outdoor_living'
    software = 'software'
    sports = 'sports_&_outdoors'
    tools = 'tools_&_hardware'
    toys = 'toys_&_games'
    video = 'video'
    
    def getDatasetPath(self):
        if (socket.gethostname() == 'InbarPC'):
            return('C:/Users/inbar/Documents/Modern Statistical Data Analysis/sorted_data/')
        if (socket.gethostname() == 'Shay-Lenovo-Lap'):
            return('C:/OrZuk/Data/sorted_data/')
        return None
        
    def getDomainPath(self):
            return self.getDatasetPath() + self.value + '/'
            
    def getBalancedFileFullPath(self):
        return self.getDomainPath() + 'processed.review.balanced'
            
    def getTrainFileFullPath(self):
        return self.getDomainPath() + 'processed.review.trainset'
            
    def getTestFileFullPath(self):
        return self.getDomainPath() + 'processed.review.testset'
    