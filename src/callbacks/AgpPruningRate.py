class AgpPruningRate(object):
    def __init__(self, initial_sparsity, final_sparsity, 
                 starting_epoch, ending_epoch, freq):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.freq = freq
        self.current_sparsity = 0
    
    def __call__(self, current_epoch):
        span = ((self.ending_epoch - self.starting_epoch) // self.freq) * self.freq
        target_sparsity = (self.final_sparsity + 
                           (self.initial_sparsity - self.final_sparsity) *
                           (1.0 - ((current_epoch - self.starting_epoch)/span))**3)
        

        if current_epoch < self.ending_epoch and current_epoch >= self.starting_epoch and current_epoch % self.freq == 0:
            if self.current_sparsity > 0:
                pct_to_prune = (target_sparsity - self.current_sparsity) / (1 - self.current_sparsity)
            else:
                pct_to_prune = target_sparsity
        
            self.current_sparsity = target_sparsity

            return  pct_to_prune
        else:
            return 0