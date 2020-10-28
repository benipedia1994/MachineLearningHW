
def binary_conv(binaryarray):
	total = 0
	arraylen = len(binaryarray)
	for i in range(0,arraylen):
		if(binaryarray[arraylen-i-1]!=1 and binaryarray[arraylen-i-1] !=0):
			print("array not binary")
			return
		total = total + (2**i)*binaryarray[arraylen-i-1]

	print("total is:" + str(total))
	return


binary_conv([0,1,0,1])
binary_conv([1,1,1,1,1])