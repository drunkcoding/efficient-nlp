import torch
from ..evaluation.criterion import ECELoss

def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = temperature.unsqueeze(
        1).expand(logits.size(0), logits.size(1))
    return logits / temperature

def temperature_scaling(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=500)
    outputs = outputs.to(device)
    labels = labels.to(device)

    nll_criterion = torch.nn.CrossEntropyLoss().to(device)
    ece_criterion = ECELoss().to(device)

    before_temperature_nll = nll_criterion(outputs, labels).item()
    before_temperature_ece = ece_criterion(outputs, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(outputs, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    after_temperature_nll = nll_criterion(temperature_scale(outputs, temperature), labels).item()
    after_temperature_ece = ece_criterion(temperature_scale(outputs, temperature), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return temperature.cpu().detach()
