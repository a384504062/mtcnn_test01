import numpy as np
#
# category_=torch.Tensor([[0,1,1,1,1,1,0,2,2,0,2,0,1,2]])
# _output_category=torch.Tensor([[1,1,2,1,1,1,0,0,0,0,2,1,2,1]])
#
# category_mask = torch.lt(category_, 2)
# print(category_mask)
#
# category = torch.masked_select(category_, category_mask)
# print(category)
#
# output_category = torch.masked_select(_output_category, category_mask)
# print(output_category)

a=np.array([[6,2,7,4,5]])
b=np.array([[3,2,2,1,4]])
# print(a[:,0])
# print(a[:][1])

arae=(a[:,2]-a[:,0])*(a[:,3]-a[:, 1])
print(a.shape)