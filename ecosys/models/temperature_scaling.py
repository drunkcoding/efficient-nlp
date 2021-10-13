import torch
from torch import nn, optim

from tqdm import tqdm

from .criterion import ECELoss

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def set_logger(self, logger):
        self.logger = logger

    # def forward(self, input_ids, attention_mask):
    #     pred = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     logits = pred.logits
    #     return self.temperature_scale(logits)

    def forward(self, input):
        output = self.model(**input)
        if isinstance(output, torch.Tensor):
            logits = output
        else:
            logits = output.logits
        return self.temperature_scale(logits)

    # def forward(self, input):
    #     logits = self.model(input)
    #     return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # for input, label in tqdm(valid_loader, desc="Training"):
        #     input = input.cuda()
        #     # print(input, label)
        #     logits = self.model(input)
        #     print(logits.shape, np.count_nonzero(logits.cpu()), logits)
        #     break

        self.cuda(1)
        nll_criterion = nn.CrossEntropyLoss().cuda(1)
        ece_criterion = ECELoss().cuda(1)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            # num_batch = 0
            # for data_batch in tqdm(valid_loader, desc="Training"):
            #     input_ids = data_batch['input_ids'].cuda()
            #     attention_mask = data_batch['attention_mask'].cuda()
            #     label = data_batch['labels'].cuda()
            #     preds = self.model(input_ids=input_ids, attention_mask=attention_mask)
            #     logits = preds.logits

            # for input, label in tqdm(valid_loader, desc="Training"):
            #     input = input.cuda()
            #     # print(input, label)
            #     logits = self.model(input)
            #     # print(logits.shape, np.count_nonzero(logits.cpu()), logits)
            
            for input, label in tqdm(valid_loader, desc="Training"):
                # print(input['pixel_values'].shape)
                output = self.model(**input)
                if isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output.logits

                logits_list.append(logits)
                labels_list.append(label)

                # num_batch += 1
                # if num_batch > 10: break
            logits = torch.cat(logits_list).cuda(1)
            labels = torch.cat(labels_list).cuda(1)
        labels = torch.flatten(labels)
        # Calculate NLL and ECE before temperature scaling
        print(logits.shape, labels.shape)
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        self.logger.info('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        # for _ in tqdm(range(10)):
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        self.logger.info('Optimal temperature: %.3f' % self.temperature.item())
        self.logger.info('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


