import torch
import torch.optim as optim
from tqdm import tqdm

from .criterion import ECELoss

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # m.weight.data.copy_(torch.eye(100))
        torch.nn.init.xavier_uniform(m.weight)
        # m.bias.data.fill_(0.01)

class ModelWithCalibration(torch.nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, out_size=2):
        super(ModelWithCalibration, self).__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.01)
        self.scale = torch.nn.Parameter(torch.ones(1) * 0.01)
        self.g_layer = torch.nn.Sequential(
            torch.nn.Linear(out_size, 100, bias=False),
            torch.nn.Linear(100, out_size, bias=False),
        )

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        scale = self.scale.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature + scale

    def calibrate(self, valid_loader):
        self.cuda()
        nll_criterion = torch.nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # optimizer = optim.SGD(self.g_layer.parameters(), lr=0.00001, momentum=0.9)

        # self.model.eval()
        # for epoch in tqdm(range(10), desc="Training"):
        #     for input, label in valid_loader:
        #         optimizer.zero_grad()
        #         with torch.no_grad():
        #             output = self.model(**input)
        #             if isinstance(output, torch.Tensor):
        #                 logits = output
        #             else:
        #                 logits = output.logits
        #         logits = self.g_layer(logits)
        #         # print(logits.shape, label.shape)
        #         loss = nll_criterion(logits, torch.flatten(label))
        #         loss.backward()
        #         optimizer.step()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, desc="Training"):
                output = self.model(**input)
                if isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output.logits
                logits = self.g_layer(logits)

                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
        labels = torch.flatten(labels)
        # Calculate NLL and ECE before temperature scaling
        print(logits.shape, labels.shape)
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        self.logger.info('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature, self.scale], lr=0.01, max_iter=500)

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
        self.logger.info('Optimal scale: %.3f', self.scale.item())
        self.logger.info('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    def forward(self, input):
        output = self.model(**input)
        if isinstance(output, torch.Tensor):
            logits = output
        else:
            logits = output.logits
        return self.temperature_scale(self.g_layer(logits))

    def set_logger(self, logger):
        self.logger = logger

