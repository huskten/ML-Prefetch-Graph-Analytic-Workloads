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

access_cache = Cache(int(2.1e+6))
with open('newtraces/pr.g16.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # row = row[0].split(',')
            pc, page, offset = int(row[0], 16), int(row[1], 16)>>12, (int(row[1], 16)>>6)&0x3f
            # print('Testing data input shapes and values... 2/14')
            # print(pc)
            # print(page)
            # print(offset)
            # print('End of testing 2/14')
            addr = int(row[1],16) # access outside of prefetcher

            access_cache.access(addr)
        
            #print(access_cache.cache[:100])
            #print("\n")
            # print(predict_LRU)
            # print(main_cache)
            # print(access_LRU)
            # print(access_cache)

            if (access_cache.hit+access_cache.miss+access_cache.no_access) % 100 == 0:
                print ('Access Hit: {}, Access Miss: {}, Access Hit Rate: {:.2f}%'.format(access_cache.hit, access_cache.miss, 100.0*access_cache.hit/float(access_cache.hit+access_cache.miss))) 
            if (access_cache.hit+access_cache.miss+access_cache.no_access) % 100000 == 0:
                with open("test_access_caches_output.txt", "a") as fa:
                    print('Access Hit: {}, Access Miss: {}, Access Hit Rate: {:.2f}%'.format(access_cache.hit, access_cache.miss, 100.0*access_cache.hit/float(access_cache.hit+access_cache.miss)), file = fa) 
        
            # pass addresses to the prefetcher
            # check if current addr hits on a prefetch

