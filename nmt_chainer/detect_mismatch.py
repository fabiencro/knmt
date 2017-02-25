import sys, os

a=open(sys.argv[1])
b=open(sys.argv[2])
counter = 0
for line1 in a:
	counter += 1
	line1 = line1.strip()
	line2 = b.readline().strip()
	print "line: ", counter, "-----", "".join(line1[0:3]), "\t", "".join(line2[0:3])
	if "".join(line1[0:3]) != "".join(line2[0:3]):
		raw_input("Press Enter!")