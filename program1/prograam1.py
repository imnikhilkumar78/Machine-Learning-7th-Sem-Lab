
import csv
a=[]
with open('finds.csv') as csfile:
    reader = csv.reader(csfile)
    for row in reader:
        a.append(row)
        print(row)
num_attributes=len(a[0])-1

print("The most general hypothesis:",["?"]*num_attributes)
print("The most specific hypothesis:",["0"]*num_attributes)

hypothesis=a[0][:-1]
print("\n The hypothesis are:")
for i in range (len(a)):
    if a[i][num_attributes] == "Yes":
        for j in range(num_attributes):
            if a[i][j]!=hypothesis[j]:
                hypothesis[j]='?'
    print( i+1,"=",hypothesis)
print("\n Final hypohthesis ")
print(hypothesis)

