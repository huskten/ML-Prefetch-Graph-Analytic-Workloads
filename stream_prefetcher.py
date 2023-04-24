import numpy as np
import csv

class Cache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = [None]*cache_size
        self.LRU_Counter = [0]*cache_size
        self.time = 0
        self.hit = 0
        self.miss = 0
        self.no_access = 0

    def add(self, key):
        key = key>>6
        self.time +=  1
        if key in self.cache:
            ind = self.cache.index(key)
            self.LRU_Counter[ind] = self.time
        elif key == 0:
            self.no_access += 1
        else:
            if None not in self.cache:
                min_ind = self.LRU_Counter.index(min(self.LRU_Counter))
                self.cache[min_ind] = key
                self.LRU_Counter[min_ind] = self.time
                
                # print(min_ind)
            else:
                self.cache[self.cache.index(None)] = key
                ind = self.cache.index(key)
                self.LRU_Counter[ind] = self.time

    def access(self, key):
        key = key>>6
        self.time +=  1
        if key in self.cache:
            ind = self.cache.index(key)
            self.LRU_Counter[ind] = self.time
            self.hit += 1
        elif key == 0:
            self.no_access += 1
        else:
            self.miss += 1
            if None not in self.cache:
                min_ind = self.LRU_Counter.index(min(self.LRU_Counter))
                self.cache[min_ind] = key
                self.LRU_Counter[min_ind] = self.time
                
                # print(min_ind)
            else:
                self.cache[self.cache.index(None)] = key
                ind = self.cache.index(key)
                self.LRU_Counter[ind] = self.time


class StreamPrefetcher:
    def __init__(self, pattern_length):
        self.pattern_length = pattern_length
        self.accesses = []

    def load(self, address):
        self.accesses.append(address)
        if len(self.accesses) >= self.pattern_length:
            pattern = self.accesses[-self.pattern_length:]
            if all(pattern[i] + (pattern[i+1]-pattern[i])/2 == pattern[i+1] for i in range(len(pattern)-1)):
                prefetch_address = pattern[-1] + (pattern[1] - pattern[0])
                cache.access(prefetch_address)

prefetcher = StreamPrefetcher(pattern_length=8)
cache = Cache(int(2.1e+6))
access_cache = Cache(int(2.1e+6))

with open('newtraces/pr.g19_segment2.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # row = row[0].split(',')
            pc, address = int(row[0], 16), (int(row[1], 16))
            address = address>>6

            if address in cache.cache:
                cache.hit += 1
            else:
                cache.miss += 1

            prefetcher.load(address)
            #prefetcher.prefetch(cache, address)
            access_cache.access(address)
            if (cache.hit+cache.miss+cache.no_access) % 100000 == 0:
                print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(cache.hit, cache.miss, 100.0*cache.hit/(cache.hit+cache.miss)))
               # print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(cache.hit, cache.miss, 100.0*cache.hit/(cache.hit+cache.miss)))
                print(cache.cache[:10])
                with open("out/streamPrefetcher_output.txt", "a") as f:
                    print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(cache.hit, cache.miss, 100*cache.hit/(cache.hit+cache.miss)), file = f)

                with open("out/streamAccess_output.txt", "a") as f1:
                    print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(access_cache.hit, access_cache.miss, 100*access_cache.hit/(access_cache.hit+access_cache.miss)), file = f1)

#print(cache.cache)
print(cache.hit + cache.miss)


print ('Hit: {}, Miss: {}, Hit Rate: {:.2f}%'.format(cache.hit, cache.miss, 100.0*cache.hit/(cache.hit+cache.miss)))
