uw_register = ['0', '0', '0', '0', '0', '0', '0', '0']

for i in range(25):
    # Insert new value at the front
    uw_register.insert(0, str(i))  
    # Remove the last value to maintain the length
    uw_register.pop()  

print(uw_register)
