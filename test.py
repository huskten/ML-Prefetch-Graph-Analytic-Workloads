import sys

# Cache_size = sys.argv[1]

# # initialize the arrays
# # note that "None" means the item is unoccupied.
# cache = [None]*cache_size
# lru_counter = [0]*cache_size  
# time=0

# for line in file:
#    arddr = parsefiel() # use the parsing.
#   block = getBlock(addr)
#  # search the cache for an empty block.
#  # fill this in.

#   # if no empty block, then find the index of the minimum
#   min_ind = index(min(lru_counter))
#   cache[min_ind] = block
#   lru_counter[min_ind] = time 

cache_size = int(sys.argv[1], 16)
cache = [None]*cache_size
LRU_counter = [0]*cache_size


def load_address(inputfilename):
    time = 0
    with open(inputfilename) as file:
        for line in file:
            pos, block = line.split(',')
            block = (int(block, base=16))<<6
            time = time + 1
            if None not in cache:
                min_ind = LRU_counter.index(min(LRU_counter))
                cache[min_ind] = block
                LRU_counter[min_ind] = time
                # print(min_ind)
            else:
                cache[cache.index(None)] = block

load_address('test.out')



print(cache)
print(LRU_counter)


