import torch
from einops import rearrange, einsum
import unittest
import numpy as np
import copy

# A'PA with diagonal A
def compute_t2(P,A):
    return einsum(P,A,A, "m n1 n2, m n1, m n2 -> m n1 n2") #  A'PA

def compute_t3(P,A,B,R):
    return torch.zeros_like(P)

class TestControl(unittest.TestCase):
    def test_P_t2(self):
        d_inner = 5
        d_state = 2
        A = torch.zeros([d_inner, d_state])
        P = torch.zeros([d_inner, d_state, d_state])

        A[0,0] = 1.0
        A[0,1] = 2.0
        A[1,0] = 3.0
        A[1,1] = -1.0
        A[3,0] = -2.5
        A[3,1] = 3.5
        P[1,0,0] = 0.5
        P[1,1,0] = 1.0
        P[1,1,1] = 0.1
        P[3,0,0] = -1.0
        P[3,0,1] = 2.0
        P[3,1,0] = 4.0
       
        # Convert A to non-compressed representation.
        Anaive = torch.zeros([d_inner, d_state, d_state])
        for i in range(A.shape[0]):
            Anaive[i,:,:] = torch.diag(A[i,:])

        t2_naive = torch.zeros(d_inner, d_state, d_state)

        # Naively compute t2: A'PA
        for i in range(A.shape[0]):
            asp = Anaive[i,:,:]
            assert len(asp.shape) == 2
            asp_trans = torch.transpose(asp, dim0=0, dim1=1)
            ps = P[i,:,:]
            t2sp =  asp_trans @ ps @ asp
            t2_naive[i,:,:] = t2sp

        t2 = compute_t2(P=P,A=A)
        assert t2.shape == t2_naive.shape
        for i in range(A.shape[0]):
            # print('t2:')
            # print(t2[i,:,:])
            # print('t2_naive:')
            # print(t2_naive[i,:,:])
            np.testing.assert_allclose(actual = t2[i,:,:], desired=t2_naive[i,:,:])

        print('t2')
        print(t2)

    def test_P_t3(self):
        d_inner = 5
        d_state = 2
        A = torch.zeros([d_inner, d_state])
        P = torch.zeros([d_inner, d_state, d_state])
        B = torch.zeros([d_state,1])
        R = torch.ones([1])
        B = torch.zeros([d_inner, d_state])

        A[0,0] = 1.0
        A[0,1] = 2.0
        A[1,0] = 3.0
        A[1,1] = -1.0
        A[3,0] = -2.5
        A[3,1] = 3.5
        P[1,0,0] = 0.5
        P[1,1,0] = 1.0
        P[1,1,1] = 0.1
        P[3,0,0] = -1.0
        P[3,0,1] = 2.0
        P[3,1,0] = 4.0
        B[1,0] = 0.9
        B[1,1] = -1.2
        B[3,0] = 0.8
        B[3,1] = 0.95

        naive_t3 = torch.zeros([d_inner,d_state,d_state])

        # Convert A to non-compressed representation.
        Anaive = torch.zeros([d_inner, d_state, d_state])
        Anaive_t = torch.zeros([d_inner, d_state, d_state])
        Bnaive = torch.zeros([d_inner, d_state, 1])
        Bnaive_t = torch.zeros([d_inner, 1, d_state])
        for i in range(A.shape[0]):
            Anaive[i,:,:] = torch.diag(A[i,:])
            Anaive_t[i,:,:] = torch.transpose(Anaive[i,:,:], 0,1)
            Bnaive[i,:,0] = B[i,:]
            Bnaive_t[i,0,:] = B[i,:]

        for i in range(A.shape[0]):
            atpb = Anaive_t[i,:,:] @ P[i,:,:] @ Bnaive[i,:,:]
            bpbr = Bnaive_t[i, :, :] @ P[i,:,:] @ Bnaive[i,:,:] + R
            print(bpbr.shape)
            assert bpbr.shape[0] == 1
            assert bpbr.shape[1] == 1
            btpa = Bnaive_t[i,:,:] @ P[i,:,:] @ Anaive[i,:,:]
            naive_t3_mat = atpb @ torch.inverse(bpbr) @ btpa
            
            naive_t3[i,:,:] = naive_t3_mat

        print('naive t3:')
        print(naive_t3)

            


           
# class LayerControl(object):
#     def __init__(self, d_inner, d_state):
#         self.Q = None 
#         self.R = 1.0
#         self.P = None
#         self.d_inner = d_inner
#         self.d_state = d_state
#         self.opt_horizon = 10

#     def ComputeP(self, A, B):
#         assert A.shape[0] == self.d_inner # D
#         assert A.shape[1] == self.d_state # N
#         if self.Q is None:
#             self.Q = torch.ones_like(A) # store one diagonal for each d_inner, D x N
#         if self.P is None:
#             self.P = copy.copy(self.Q)

#         P = self.P

#         # for i in range(self.opt_horizon):
#         #     # Implement dynamic Discrete Ricatti Algebraic Equation
#         #     # P = Q + A'PA - A'PB((B'PB+R)^-1)B'PA
#         #     t1 = self.Q # D N
#         #     t2 = 
                        

# class MambaControl(object):
#     def __init__(self, n_layer):
#         self.layers = list()
#         for i in range(0,n_layer):
#             self.layers.append(LayerControl())


#     # Return: effort, B x D
#     def control_callback(dA, dB, C, D, ssm_state, layer_idx)
#         # squeeze dim[0]/batch index from dA, dB, C            
        
#         self.P = self.ComputeP(Q,R,dA,dB,C,D)
#         return effort


if __name__ == '__main__':
    unittest.main()

