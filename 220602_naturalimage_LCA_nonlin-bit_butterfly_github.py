from numpy import *
import copy
import matplotlib.pyplot as plt
from images_sim_utils import *
# get_ipython().magic('matplotlib inline')


#######################################################################
# parameters #
case = 'LCA'  # LCA
col_num = 98    # dictionary number
alpha_range = [0.0004]  # learning rate 0.0004
Oja_iter = 50

p_nonlinear = [0]     #[0,0.89,13.32,2.75]   ideal, Ti, pristine, densified
d_nonlinear = [0]     #[0,1.38,18.93,2.92]
w_n_state = [0]   #[0,0,200,200]   # 2**6  , state == 0 is full analog case

w_max = 1   # 0.5

#######################################################################
def implot(image):
	plt.imshow(image, cmap = 'gray', interpolation = 'none')
	plt.axis('off')

def gen_input_patch():    # generate input for LCA
	global patch_size, patch_count,input_image, scale_patch
	scale_patch = 256   #256
	patch_size = (7,7)
	input_image = plt.imread('butterfly_140.pgm').astype(float64)   #  'images/lena140.pgm'
	patch_count = array(input_image.shape)/array(patch_size)
	input_patch = rollaxis(input_image.reshape(int(patch_count[0]),int(patch_size[0]),int(patch_count[1]),int(patch_size[1])),2,1)
	input_patch = input_patch.reshape(-1,product(patch_size)).T
	return input_patch/scale_patch   #49*400

def gen_dictionary():    
	# scale_Dic = 1    #1e4 for SPICE sim.
	Dic = loadtxt('butterfly_weight98_state-'+str(n_state)+'_nonP-'+str(non_p)+'_nonD-'+str(non_d)+'_alpha-'+str(alpha)+'.txt')  
	return Dic    #49*98
#######################################################################
def soft_threshold(u,lamda):
	soft_threshold_a = .75
	soft_threshold_m = 1/(1-soft_threshold_a)
	t = u.copy()

	b = lamda*(1-soft_threshold_m)

	soft = where(abs(t)<lamda)
	u_soft = u[soft]
	t[soft] = soft_threshold_m*u_soft + sign(u_soft)*b 

	t[abs(u)<soft_threshold_a*lamda] = 0

	return t

def hard_threshold(u,lamda):
	a = u.copy()
	a[abs(a)<lam] = 0
	
	return a
#######################################################################
def LCA(input_patch,Dic):   # input_patch : 49x400, Dic : 49x98
	global lamda, avgL0
	patch_num = input_patch.shape[1]  #400
	iteration = 5000  #1500
	lamda = 0.02    #0.1
	tau = 0.008  #0.008
	save_patch_num = 0  # select patch number for saving mem. pot

	u = zeros((patch_num,Dic.shape[1]))   # mem potential, initial value,  400x98
	a = zeros((patch_num,Dic.shape[1]))   # activity, initial value,  400x98
	u_save = zeros((Dic.shape[1],iteration+1))    # save mem. pot. of one patch for plot,  98x5000

	for i in range(iteration):
		a = soft_threshold(u,lamda).T    # 98*400
		residue = input_patch.T - Dic.dot(a).T   # 400x49
		dot = residue.dot(Dic)   # 400x98

		dudt = tau*(-u + dot + a.T)    # 400x98
		u += dudt

		u_save[:,i] = u[save_patch_num,:]
		# print(i, ((residue)**2).mean())
		# print ("iteration:%d/%d" %(i,iteration))

	mse = ((residue)**2).mean()
	avgL0 = count_nonzero(a)/a.shape[1]
	# savetxt("mem_pot.txt",u_save)

	return a
#######################################################################
def MSE(input_patch,Dic,a_total):
	# MSE = linalg.norm(input_patch - Dic.dot(a_total*scale_Dic)) / input_patch.shape[0]
	MSE = ((input_patch - Dic.dot(a_total))**2).mean()
	print('MSE : '+str(MSE))
	print('avgL0 : '+str(avgL0))
	return MSE

def reconstruct(Dic,a_total,MSE):
	recon_image = rollaxis(Dic.dot(a_total).T.reshape(int(patch_count[0]),int(patch_count[1]),int(patch_size[0]),int(patch_size[1])),2,1) #20x7x20x7
	recon_image = recon_image.reshape(int(patch_count[0])*int(patch_size[0]),int(patch_count[1])*int(patch_size[1]))
	plt.figure(figsize=(10, 10))
	plt.subplot(1,2,1)
	plt.title("original")
	implot(input_image)
	plt.subplot(1,2,2)
	plt.title("reconstructed\n MSE = %f" % (MSE))
	plt.title('$\lambda$={:.2f} $L_0$={:.2f}\n MSE={:.3e}'.format(lamda,avgL0,MSE))
	implot(recon_image)
	plt.savefig('butterfly_reconstructed_state-'+str(n_state)+'_nonP-'+str(non_p)+'_nonD-'+str(non_d)+'_alpha-'+str(alpha)+'.png') ; plt.close('all')
	# plt.show()
	return recon_image
#######################################################################
def learning(patches, col_num, alpha, weight):
	# sample_count = patches.shape[0]
	
	train_count = np.zeros(col_num);
	#weight = np.random.normal(0,1,10*W.shape[1]).reshape([-1,W.shape[1]])
	#weight = np.zeros([10,W.shape[1]])
	for i, x in enumerate(patches):
		# weight = weight/weight.sum(axis=1)[:,np.newaxis]
		y = tensordot(x,weight, (0,0))
		index = np.argmax(y)
		# index = eps_greedy(1,index,col_num)
		train_count[index] += 1
		residual = x-y[index]*weight[:,index]  
		#print y[index]*residual 

		###Oja's rule 
		# weight[:,index] += alpha*y[index]*residual
		before = weight[:,index]
		after = weight[:,index] + alpha*y[index]*residual
		fin_w = calc_final_w(before, after, n_state)
		weight[:,index] = fin_w
		
	return weight, train_count   # weight.T = 49x98

######################################################################
def calc_time_nonlinear_p(wei):
    if non_p != 0:
        # G_1 = (G_LRS - G_HRS_p) / (1-np.exp(-non_p))
        # tprog = np.log( 1-(wei*(G_LRS-G_HRS_p)/G_1) ) / (-non_p)
        tprog = np.log(1-(wei*(1-np.exp(-non_p)))) / (-non_p)
    else:
        tprog = wei     # perfect linear
    return tprog

def calc_time_nonlinear_n(wei):
    if non_d != 0:
        # G_1 = (G_LRS - G_HRS_d) / (1-np.exp(-non_d))
        # tprog = 1 + (np.log(1-((G_LRS-G_HRS_d)*(1-wei)/G_1)) / non_d)
        tprog = 1 + ( np.log( 1-((1-wei)*(1-np.exp(-non_d))) ) / non_d)
    else:
        tprog = wei     # perfect linear
    return tprog

def calc_w_nonlinear_p(t):
    if non_p != 0:
        # G_1 = (G_LRS - G_HRS_p) / (1-np.exp(-non_p))
        # wei = (((G_1*(1-np.exp(-non_p*t))+G_HRS_p) - G_HRS_p) / (G_LRS - G_HRS_p))
        wei = (1-np.exp(-non_p*t)) / (1-np.exp(-non_p))
    else:
        wei = t     # perfect linear
    return wei

def calc_w_nonlinear_n(t):
    if non_d != 0:
        # G_1 = (G_LRS - G_HRS_d) / (1-np.exp(-non_d))
        # wei = ( ((G_LRS - G_1*(1-np.exp(-non_d*(1-t)))) - G_HRS_d) / (G_LRS - G_HRS_d) )
        wei = 1 - ((1-np.exp(-non_d*(1-t))) / (1-np.exp(-non_d)))
    else:
        wei = t     # perfect linear
    return wei

def calc_digit_t_p(time_w_p, n_state):
    ### digitize time for positive area
    if n_state != 0:
        # n_state = 2**bit
        bin_p = np.linspace(0, w_max, n_state)    # bin_p = np.linspace(0, 1, n_state)
        # print(bin_p)
        # sys.exit(1)
        calc_time_p = np.digitize(time_w_p, bin_p)
        calc_time_p = (calc_time_p-1) / (n_state-1)

        triger = bin_p[1]-bin_p[0]   #!!!!!!!!!!!!!!!!!!

        return calc_time_p

    else:
        return time_w_p

def calc_digit_t_n(time_w_n, n_state):
    ### digitize time for negative area
    if n_state != 0:
        # n_state = 2**bit
        bin_n = np.linspace(w_max, 0, n_state)
        calc_time_n = np.digitize(time_w_n, bin_n) + (n_state) 
        calc_time_n = (((2*n_state)-1) - calc_time_n) / (n_state-1)
        return calc_time_n

    else:
        return time_w_n
######################################################################
def calc_final_w(bef_W1,aft_W1,state):
    delta_w1 = np.abs(aft_W1) - np.abs(bef_W1)

    mask_p1 = delta_w1 >= 0; mask_n1 = delta_w1 < 0; #mask_zero1 = delta_w1 == 0;

    # delta_w1_p = delta_w1 * mask_p1; delta_w1_n = delta_w1 * mask_n1; #delta_w1_zero = delta_w1 * mask_zero1;

    # delta_time_w1_p = delta_w1_p # tprog with linear case is delta w itself
    # delta_time_w1_n = delta_w1_n
    # # delta_time_w1_zero = delta_w1_zero

    del_time_w1 = delta_w1   # tprog with linear case is delta w itself

    # if state != 0:     # for limited state number
    #     del_time_w1_c = np.copy(del_time_w1)

    #     gap_index = np.linspace(0, w_max, state)      # gap_index = np.linspace(0, 1, state)
    #     gap = (gap_index[1] - gap_index[0])/2

    #     del_time_w1[del_time_w1_c>0] += gap
    #     del_time_w1[del_time_w1_c>gap] -= gap

    #     del_time_w1[del_time_w1_c<0] -= gap
    #     del_time_w1[del_time_w1_c<gap] += gap

    ###########################################################################

    origin_time_w1_p = calc_time_nonlinear_p(np.abs(bef_W1))
    origin_time_w1_n = calc_time_nonlinear_n(np.abs(bef_W1))

    total_time_w1_p = origin_time_w1_p + del_time_w1
    total_time_w1_n = origin_time_w1_n + del_time_w1

    total_time_w1_p = np.clip(total_time_w1_p, 0, 1)
    total_time_w1_n = np.clip(total_time_w1_n, 0, 1)

    digitize_time_w1_p = calc_digit_t_p(total_time_w1_p, state) # print(digitize_time_w1_p, '\n')
    digitize_time_w1_n = calc_digit_t_n(total_time_w1_n, state) # print(digitize_time_w1_n, '\n')


    fin_w1_p = calc_w_nonlinear_p(digitize_time_w1_p)
    fin_w1_n = calc_w_nonlinear_n(digitize_time_w1_n)
    fin_w1_p[~mask_p1] = 0
    fin_w1_n[~mask_n1] = 0
    fin_w1 = fin_w1_p + fin_w1_n
    fin_w1 = fin_w1*np.sign(aft_W1)
    fin_w1 = fin_w1

    return fin_w1
#######################################################################
# main #
print("simulation started!!")

for n_state, non_p, non_d in zip(w_n_state, p_nonlinear, d_nonlinear):
	for alpha in alpha_range:
		if case == 'LCA':
			# run LCA
			a = gen_input_patch()
			b = gen_dictionary()
			a_total = LCA(a,b)
			c = MSE(a,b,a_total)
			d = reconstruct(b,a_total,c)

print("Done!!")






