'''
Created on 13 de dez de 2017

@author: luisfantozzialvarez
'''
from test.test_socketserver import ForkingErrorTestServer

'''
Solves Ulyssea's  (2010) model including FGTS
'''
import numpy as np
from scipy.optimize import fsolve

class Ulyssea(object):
    '''
    classdocs
    '''


    def __init__(self, tau_pi, tau_omega, r, phi_f, phi_i, s_i, s_f, rho, K_f, K_i, b, eta, a, A, delta):
        '''
        Constructor: Initialises economy with parameters 
        tau_pi: payroll tax, 
        tau_omega: labour tax, 
        r: discount rate, 
        phi_f: formal bargain power, 
        phi_i: informal bargain power, 
        s_i: informal separation rate, 
        s_f: formal separation rate, 
        rho: CES elasticity parameter, 
        K_f: formal sector's entry cost, 
        K_i: informal sector's entry cost, 
        b: unemployment benefit (expected discounted value), 
        eta: matching elasticity, 
        a: formal sector's relevance, 
        A: matching scale parameter
        '''
        self.tau_pi, self.tau_omega, self.r, self.phi_f, self.phi_i, self.s_i, self.s_f, self.rho, self.K_f, self.K_i, self.b, self.eta, self.a,\
         self.A, self.delta = tau_pi, tau_omega, r, phi_f, phi_i, s_i, s_f, rho, K_f, K_i, b, eta, a, A, delta
     
    """
    Arrival rate of offers for worker. Depends on theta_j, labour tightness in sector j
    """ 
    def Lambda(self, theta):
        return  self.A*theta**(1-self.eta)
    
    """
    Arrival rate for firm. Depends on theta_j, labour tightness in sector j
    """ 
    def Q(self, theta):
        return self.A*theta**(-self.eta)
    
    """
    Equilibrium equations that solve the model
    """
    def Eq_Conditions(self, endog):
        gamma = endog[0]
        theta_f = endog[1]
        theta_i = endog[2]
        rU = (self.phi_f/(1- self.phi_f))*self.r*self.K_f*theta_f + (self.phi_i/(1- self.phi_i))*self.r*self.K_i*theta_i
        pF1 = self.a*gamma**(self.rho -1)*(self.a*gamma**self.rho + (1-self.a)*(1-gamma)**self.rho)**((1-self.rho)/self.rho)
        pI1 = (1-self.a)*(1-gamma)**(self.rho - 1)*(self.a*gamma**self.rho + (1-self.a)*(1-gamma)**self.rho)**((1-self.rho)/self.rho)
        pF2 = ((1+ self.tau_pi)/(1-(1-self.delta*self.s_f)*self.tau_omega))*(((self.phi_f*(self.r + self.s_f))/((1-self.phi_f)*self.Q(theta_f)))*self.r*self.K_f \
            + rU - self.s_f*self.b) + (self.r + self.s_f + self.Q(theta_f))*self.r*self.K_f/self.Q(theta_f)
        pI2 = ((self.r + self.s_i)/((1- self.phi_i)*self.Q(theta_i)))*self.r*self.K_i + rU  + self.r*self.K_i
        gamma_flow = (self.s_i*self.Lambda(theta_f))/(self.s_i*self.Lambda(theta_f) + self.s_f*self.Lambda(theta_i))
        
        return np.array([pF1 - pF2, pI1 - pI2, gamma - gamma_flow])
    
    #Solves model for given starting values of endog to fsolve
    def SolveModel(self, start):
        solution = fsolve(self.Eq_Conditions, start)
        gamma = solution[0]
        theta_f = solution[1]
        theta_i = solution[2]
        u = (self.s_f*gamma)/(self.s_f*gamma + self.Lambda(theta_f))
        rU = (self.phi_f/(1- self.phi_f))*self.r*self.K_f*theta_f + (self.phi_i/(1- self.phi_i))*self.r*self.K_i*theta_i
        
        wF = (1/(1-(1-self.delta*self.s_f)*self.tau_omega))*(((self.phi_f*(self.r + self.s_f))/((1-self.phi_f)*self.Q(theta_f)))*self.r*self.K_f \
            + rU - self.s_f*self.b)
        
        wI = (self.phi_i/(1-self.phi_i))*((self.r + self.s_i)/self.Q(theta_i))*self.r*self.K_i + rU 
        
        diff = (wF - wI)/wI
        
        pF = self.a*gamma**(self.rho -1)*(self.a*gamma**self.rho + (1-self.a)*(1-gamma)**self.rho)**((1-self.rho)/self.rho)
        pI = (1-self.a)*(1-gamma)**(self.rho - 1)*(self.a*gamma**self.rho + (1-self.a)*(1-gamma)**self.rho)**((1-self.rho)/self.rho)
        
        W = (1-u)*(gamma*(pF - self.r*self.K_f) + (1-gamma)*(pI- self.r*self.K_i)) - u*(gamma*theta_f*self.r*self.K_f + (1-gamma)*theta_i*self.r*self.K_i)

        return(np.array([1-gamma, diff, W,  (1-(1-self.delta*self.s_f)*self.tau_omega)*wF]))
    