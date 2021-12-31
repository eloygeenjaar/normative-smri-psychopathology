import torch
import gpytorch
import matplotlib.pyplot as plt
from importlib import import_module
from tqdm import tqdm


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, mean, kernel, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        if mean == 'LinearMean':
            self.mean_module = getattr(import_module(f'gpytorch.means'), mean)(1)
        else:
            self.mean_module = getattr(import_module(f'gpytorch.means'), mean)()
        if kernel == 'PolynomialKernel':
            self.covar_module = getattr(import_module(f'gpytorch.kernels'), kernel)(3)
        else:
            self.covar_module = getattr(import_module(f'gpytorch.kernels'), kernel)()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predict(model, likelihood, x_test, range_in = torch.linspace(-0.2, 1.2, 100)):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad():
        # Test points are regularly spaced along [0,1]
        range_pred = likelihood(model(range_in))
        test_pred = likelihood(model(x_test))
        return range_in.cpu().numpy(), range_pred, test_pred


def plot(x_train, y_train, y_pred, x_test=torch.linspace(-0.2, 1.2, 100)):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        # Get upper and lower confidence bounds
        lower, upper = y_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(x_test.numpy(), y_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(x_test.numpy().ravel(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()


def train(x_train, y_train,
          model, likelihood,
          x_valid=None, y_valid=None, training_iter=int(100)):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    pbar = tqdm(range(training_iter))
    for _ in pbar:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        pbar.set_description(f'{loss}')
        loss.backward()
        optimizer.step()

    if x_valid is not None and y_valid is not None:
        model.eval()
        likelihood.eval()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        with torch.no_grad():
            valid_ll = mll(model(x_valid), y_valid)

        return valid_ll


def train_test_gpr(mean, kernel,
                   x_train, x_test,
                   y_train, y_test,
                   num_epochs):
    x_train = torch.from_numpy(x_train).float().squeeze()
    x_test = torch.from_numpy(x_test).float().squeeze()

    y_train = torch.from_numpy(y_train).float().squeeze()
    y_test = torch.from_numpy(y_test).float().squeeze()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(mean, kernel, x_train, y_train, likelihood)

    likelihood.train()
    model.train()
    train(x_train, y_train,
          model, likelihood, training_iter=num_epochs)

    model.eval()
    likelihood.eval()
    range_in, range_pred, test_pred = predict(model, likelihood, x_test)
    return range_in, range_pred, test_pred, model, likelihood


def validate_gpr(mean, kernel,
                 x_train, x_valid,
                 y_train, y_valid,
                 num_epochs):
    x_train = torch.from_numpy(x_train).float().squeeze()
    x_valid = torch.from_numpy(x_valid).float().squeeze()

    y_train = torch.from_numpy(y_train).float().squeeze()
    y_valid = torch.from_numpy(y_valid).float().squeeze()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(mean, kernel, x_train, y_train, likelihood)

    likelihood.train()
    model.train()
    ll = train(x_train, y_train,
               model, likelihood,
               x_valid, y_valid,
               training_iter=num_epochs)

    return ll
