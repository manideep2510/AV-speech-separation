#pip3 install mir_eval
#target_samples and predicted_samples shoud have same size.
#input_shape--(batch_size,samples)
#returns mean_sdr,sdr_values

from mir_eval.separation import bss_eval_sources

def metric_eval(target_samples, predicted_samples):
	
	sdr_batch=[]
	
	batch_size=(target_samples.shape)[0]
	for i in range(batch_size):
		sdr, sir, sar, _ = bss_eval_sources(target_samples[i], predicted_samples[i], compute_permutation=False)
		sdr_batch.append(sdr[0])

	return sdr_batch.mean(),sdr_batch
	
