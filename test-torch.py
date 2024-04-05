import torch

print(torch.cuda.device_count())

tensor = torch.tensor([1, 2, 3])

cuda_device = torch.device("cuda")
# Check if CUDA is available
if torch.cuda.is_available():
    # Create a tensor on CPU

    tensor_cuda = tensor.to(cuda_device)

    print("Tensor on CUDA device:")
    print(tensor_cuda)
else:
    print("CUDA is not available. You may want to run this code on a machine with a GPU.")
