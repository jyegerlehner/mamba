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

# computes A'PB
def compute_atpb(A,P,B):
    return einsum(A,P,B, "i n1, i n1 n2, i n2 -> i n1")

# Computes B'PA
def compute_btpa(B,P,A):
    return einsum(B,P,A, 'i n1, i n1 n2, i n2-> i n2')

# computes B'PB+R   'ji,jk,ki->i'
def compute_btpbr(B,P,R):
    # B   D x N
    # P   D x N x N
    # R   scalar
    return einsum(B,P,B, 'i j, i j k, i k -> i') + R

def compute_t3(atpb, btpbr, btpa):
    first = einsum(atpb, btpbr.pow_(-1), 'i n, i -> i n')
    return einsum(first, btpa, 'i n1, i n2 -> i n1 n2')

def iterate_P(Q, R, A, B, P):
    atpb = compute_atpb(A=A,P=P,B=B)
    btpbr = compute_btpbr(B=B,P=P,R=R)
    btpa = compute_btpa(B=B, P=P, A=A)
    # dynamic ricatti algebraic ricatti equation
    # new P = Q + A'PA - A'PB[(B'PB+R)^-1]B'PA
    t1 = torch.diag_embed(Q) 
    t2 = compute_t2(P,A)
    t3 = compute_t3(atpb=atpb, btpbr=btpbr, btpa=btpa)
    return t1 + t2 - t3

class TestControl(unittest.TestCase):
    def test_P_t3(self):
        d_inner = 5
        d_state = 2
        A = torch.zeros([d_inner, d_state])
        P = torch.zeros([d_inner, d_state, d_state])
        B = torch.zeros([d_inner,d_state])
        R = torch.ones([d_inner])

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
        B[3,0] = -0.9
        B[3,1] = 1.05

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

        t3_atpb = compute_atpb(A,P,B)
        t3_btpbr = compute_btpbr(B,P,R)
        t3_btpbr_copy = copy.deepcopy(t3_btpbr)
        t3_btpa = compute_btpa(B,P,A)
        t3 = compute_t3(t3_atpb, t3_btpbr, t3_btpa)

        for i in range(A.shape[0]):
            atpb = Anaive_t[i,:,:] @ P[i,:,:] @ Bnaive[i,:,:]
            bpbr = Bnaive_t[i, :, :] @ P[i,:,:] @ Bnaive[i,:,:] + R[i] #
            assert bpbr.shape[0] == 1
            assert bpbr.shape[1] == 1
            btpa = Bnaive_t[i,:,:] @ P[i,:,:] @ Anaive[i,:,:]
            naive_t3_mat = atpb @ torch.inverse(bpbr) @ btpa
            naive_t3[i,:,:] = naive_t3_mat
            print('atpb shape:{0}'.format(atpb.shape))
            print('bpbr shape:{0}'.format(bpbr.shape))
            print('btpa shape:{0}'.format(btpa.shape))
            print('naive_t3 shape:{0}'.format(naive_t3.shape))
            np.testing.assert_allclose(atpb[:,0], t3_atpb[i,:])
            np.testing.assert_allclose(bpbr[0,0], t3_btpbr_copy[i])            
            assert len(btpa.shape) == 2
            assert btpa.shape[0] == 1
            assert btpa.shape[1] == d_state
            np.testing.assert_allclose(btpa[0,:], t3_btpa[i,:])

        assert len(t3_atpb.shape) == 2
        assert t3_atpb.shape[0] == d_inner
        assert t3_atpb.shape[1] == d_state
        
        assert len(t3_btpbr.shape) == 1
        assert t3_btpbr.shape[0] == d_inner

        assert len(t3_btpa.shape) == 2
        assert t3_btpa.shape[0] == d_inner
        assert t3_btpa.shape[1] == d_state

        np.testing.assert_allclose(t3, naive_t3)

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

        assert len(t2.shape) == 3
        assert t2.shape[0] == d_inner
        assert t2.shape[1] == d_state
        assert t2.shape[2] == d_state
      
    def test_P_iteration(self):
        d_inner = 5
        d_state = 2
        A = torch.zeros([d_inner, d_state])
        B = torch.zeros([d_inner,d_state])
        R = torch.ones([d_inner])
        Q = torch.ones([d_inner, d_state]) # diagonal elements of Q

        A[0,0] = 1.0
        A[0,1] = 2.0
        A[1,0] = 3.0
        A[1,1] = -1.0
        A[3,0] = -2.5
        A[3,1] = 3.5
        B[1,0] = 0.9
        B[1,1] = -1.2
        B[3,0] = -0.9
        B[3,1] = 1.05

        # Convert A to non-compressed representation.
        Anaive = torch.zeros([d_inner, d_state, d_state])
        Anaive_t = torch.zeros([d_inner, d_state, d_state])
        Bnaive = torch.zeros([d_inner, d_state, 1])
        Bnaive_t = torch.zeros([d_inner, 1, d_state])
        Rnaive = torch.zeros([d_inner,1,1])
        Qnaive = torch.zeros([d_inner, d_state, d_state])
        P1 = torch.zeros([d_inner, d_state, d_state])
        P2 = torch.zeros([d_inner, d_state, d_state])

        for i in range(A.shape[0]):
            Anaive[i,:,:] = torch.diag(A[i,:])
            Anaive_t[i,:,:] = torch.transpose(Anaive[i,:,:], 0,1)
            Bnaive[i,:,0] = B[i,:]
            Bnaive_t[i,0,:] = B[i,:]
            Rnaive[i,0,0] = R[i]
            Qnaive[i,:,:] = torch.diag(Q[i,:])
            P1[i,:,:] = torch.diag(Q[i,:])
            P2[i,:,:] = torch.diag(Q[i,:])

        P1_history = list()
        P1_history.append(copy.deepcopy(P1))
        # iterate the refinement of P 10 times.
        for iter in range(0,10):
            # Naive calculation of P iteration using matrix multiplies
            # using dynamic DARE
            for i in range(0, d_inner):
                P1[i,:,:] = Qnaive[i,:,:] + Anaive_t[i,:,:]@P1[i,:,:]@Anaive[i,:,:] - \
                            Anaive_t[i,:,:]@P1[i,:,:]@Bnaive[i,:,:]@(torch.inverse(Bnaive_t[i,:,:]@P1[i,:,:]@Bnaive[i,:,:] + Rnaive[i,:,:])@Bnaive_t[i,:,:]@P1[i,:,:]@Anaive[i,:,:])
            P1_history.append(copy.deepcopy(P1))

        P2_history = list()
        P2_history.append(copy.deepcopy(P2))
        for iter in range(0,10):
            P2 = iterate_P(Q, R, A, B, P2)
            P2_history.append(copy.deepcopy(P2))

        assert len(P1_history) == len(P2_history)
        assert len(P1_history) == 11
        for i in range(0,11):
            np.testing.assert_allclose(P2_history[i], P1_history[i], atol=0.001)

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

