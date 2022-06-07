import matplotlib.pyplot as plt
mylist = []
fig1, ax1 = plt.subplot_mosaic(mosaic = '''
                                      AA
                                      BC
                                      '''  )

fig2, ax2 = plt.subplot_mosaic(mosaic = '''
                                      AA
                                      BC
                                      '''  )

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
ax1['A'].plot(x, y)
mylist.append(fig1)
mylist.append(fig2)
plt.close()
plt.close()
myfig = plt.figure(mylist[0])
myfig.savefig('/Users/bementmbp/Desktop/test.png', )