import sys
import getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], 'h', ['batch_size=', 'epochs='])
    
    for o, a in opts:
        if o == '--batch_size':
            batch_size = int(a.strip())
        elif o == '--epochs':
            epochs = int(a.strip())
        else:
            assert False, 'Unhandled option!'
except getopt.GetoptError as err:
    print(err)
    print('Using default values')
    batch_size = 16
    epochs = 5
            
print('batch_size:', batch_size*2)
print('epochs:', epochs*7)
