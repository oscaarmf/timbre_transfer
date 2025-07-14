from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import json
import os

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        self.learning_rate = config.learning_rate 

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        self.save_step = config.save_step
        self.save_dir = config.save_dir

        self.history = {
            'G/loss_id': [],
            'G/loss_id_psnt': [],
            'G/loss_cd': []
        }

        self.initial_step = config.initial_step
        self.checkpoint_path = config.checkpoint_path

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.learning_rate)
        
        self.G.to(self.device)

        if self.checkpoint_path:
            # Load checkpoints if specified.
            if not os.path.exists(self.checkpoint_path):
                raise Exception('There is no checkpoint at {}.'.format(self.checkpoint_path))
            self.G.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device)['model'])
            print('Loaded checkpoints from {}.'.format(self.checkpoint_path))


        if self.initial_step > 0:
            # Load checkpoints.
            self.checkpoint_path = os.path.join(self.save_dir, 'G-{}.pth'.format(self.initial_step))
            if not os.path.exists(self.checkpoint_path):
                raise Exception('There is no checkpoint at {}.'.format(self.checkpoint_path))
            self.G.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print('Loaded checkpoints from {}.'.format(self.checkpoint_path))

            self.history = json.load(open(os.path.join(self.save_dir, 'history.json'), 'r'))

        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        print(f"Dataset size: {len(data_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.initial_step + 1, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)

            if x_identic.dim() == 4 and x_identic.size(1) == 1:
                x_identic = x_identic.squeeze(1)
            if x_identic_psnt.dim() == 4 and x_identic_psnt.size(1) == 1:
                x_identic_psnt = x_identic_psnt.squeeze(1)
                
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                    self.history[tag].append(loss[tag])
                print(log)
                # Save logs.
                with open('{}/history.json'.format(self.save_dir), 'w') as f:
                    json.dump(self.history, f, indent=4)

            # Save model checkpoints.
            if (i+1) % self.save_step == 0:
                torch.save(self.G.state_dict(), '{}/G-{}.pth'.format(self.save_dir, i+1))
                print('Saved model checkpoints into {}...'.format(self.save_dir))
                

    
    

    