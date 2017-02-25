import sys,os

a=open(sys.argv[1])
b=open(sys.argv[2])

c=open(sys.argv[3])
d=open(sys.argv[4])

a_ids = set()
b_ids = set()
c_ids = set()
d_ids = set()

counter = 0
for i in a:
	if counter%2 != 0:
		a_ids.add(i.strip())
	counter += 1
	
counter = 0
for i in b:
	if counter%2 != 0:
		b_ids.add(i.strip())
	counter += 1

counter = 0
for i in c:
	if counter%2 != 0:
		c_ids.add(i.strip())
	counter += 1

counter = 0
for i in d:
	if counter%2 != 0:
		d_ids.add(i.strip())
	counter += 1

#assert len(a_ids) == len(b_ids)
#assert len(c_ids) == len(d_ids)

print "Done reading"
counter = 0

#print a_ids
#print c_ids
for i in a_ids:
	if i in d_ids:
		#print i
		counter += 1
print counter