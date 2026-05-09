import numpy as np
import torch


class IntensityStatsFixer(torch.nn.Module):
    def __init__(
        self,
        target_mean_intensity=0.5,
        target_std_intensity=0.25,
        max_steps=30,
        eps=1e-5,
        lr=5.0,
    ):
        super().__init__()
        self.target_mean_intensity = target_mean_intensity
        self.target_std_intensity = target_std_intensity
        self.eps = eps
        self.last_b = None
        self.last_c = None
        self.lr = lr
        self.max_steps = max_steps

    def _inverse_sigmoid(self, im):
        return torch.log((im + self.eps) / (1 - im - self.eps))

    def _measure_mean(self, im, alpha=None, sum_alpha=None):
        # returns mean intensity per image, potentially weighted by alpha
        n = len(im)
        contract = lambda x: x.reshape(
            [n, -1]
        )  # collapse to images x pixels (2d Tensor)
        if alpha is None:
            m = contract(im).mean(dim=1, keepdims=False)
        else:
            if sum_alpha is None:
                sum_alpha = contract(alpha).sum(dim=1, keepdims=False)
            m = contract(im * alpha).sum(dim=1, keepdims=False) / sum_alpha
        return m

    def _measure_std(self, im, alpha=None, sum_alpha=None):
        # returns std intensity per image, potentially weighted by alpha
        n = len(im)
        expand = lambda x: x.reshape(
            [
                n,
            ]
            + [1] * (im.ndim - 1)
        )  # reshape to image shape
        contract = lambda x: x.reshape(
            [n, -1]
        )  # collapse to images x pixels (2d Tensor)

        if alpha is None:
            s = contract(im).std(dim=1, keepdims=False)
        else:  #
            if sum_alpha is None:
                sum_alpha = contract(alpha).sum(dim=1, keepdims=False)
            m = expand(contract(im * alpha).sum(dim=1, keepdims=False) / sum_alpha)
            v = (
                contract(torch.square(im - m) * alpha).sum(dim=1, keepdims=False)
                / sum_alpha
            )
            s = torch.sqrt(v)
        return s

    def _adjust_im(self, decompressed_im, b, c, alpha=None):
        n = len(decompressed_im)
        expand = lambda x: x.reshape(
            [
                n,
            ]
            + [1] * (decompressed_im.ndim - 1)
        )
        b = expand(b)
        c = expand(c)
        m = expand(self._measure_mean(decompressed_im, alpha=alpha))
        return torch.sigmoid((decompressed_im - m) * c + m + b)

    def forward(self, im, alpha=None):
        """normalize a face, given an optional boolean alpha map"""

        n = im.shape[0]

        # initiate scale and bias variables as zeros and ones, or use the cached results of a previous iteration
        if self.last_b is None:
            b = torch.zeros((n,), device=im.device)
        else:
            b = torch.tensor(self.last_b, device=im.device)
        if self.last_c is None:
            c = torch.ones((n,), device=im.device)
        else:
            c = torch.tensor(self.last_c, device=im.device)

        # inverse sigmoid - stretch the image to -inf ... +inf
        decompressed_im = self._inverse_sigmoid(
            im.clamp(min=self.eps, max=1.0 - self.eps)
        )

        for i in range(self.max_steps):

            adjusted_im = self._adjust_im(decompressed_im, b, c, alpha=alpha)
            m = self._measure_mean(adjusted_im, alpha=alpha)
            b = b + (self.target_mean_intensity - m) * self.lr
            # print('m=',m.mean().item(),' b=',b.mean().item(),end=' ')

            adjusted_im = self._adjust_im(decompressed_im, b, c, alpha=alpha)
            s = self._measure_std(adjusted_im, alpha=alpha)
            # print('s=',s.mean().item(),' c=',c.mean().item())
            c = c * (self.target_std_intensity / s)
        self.last_b = (
            b.detach().cpu().numpy()
        )  # store as numpy array so we have no PyTorch memory leaks when using retain_graph=True
        self.last_c = c.detach().cpu().numpy()

        adjusted_im = self._adjust_im(decompressed_im, b, c, alpha=alpha)

        if alpha is None:
            return adjusted_im
        else:
            return torch.where(alpha, adjusted_im, im)


def _test_IntensityStatsFixer():
    z = torch.tensor([0.1], device="cuda:0")
    z.requires_grad_(True)

    im = torch.rand(size=(55, 1, 224, 224), device="cuda:0")
    alpha_im = torch.rand(size=(55, 1, 224, 224), device="cuda:0") > 0.8
    im = torch.clamp(im + z, min=0.0, max=1.0)
    fixer = IntensityStatsFixer(
        target_mean_intensity=0.5, target_std_intensity=0.25, max_steps=15
    )
    im = fixer(im, alpha=alpha_im)

    loss = im.mean()
    loss.backward()
    print("z.grad=", z.grad)


if __name__ == "__main__":
    _test_IntensityStatsFixer()

# b.requires_grad_(True)

# 	n, m = 2, 3
# 	x = cp.Variable(n)
# 	A = cp.Parameter((m, n))
# 	b = cp.Parameter(m)
# 	constraints = [x >= 0]
# 	objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
# 	problem = cp.Problem(objective, constraints)
# 	assert problem.is_dpp()

# 	cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
# 	A_tch = torch.randn(m, n, requires_grad=True)
# 	b_tch = torch.randn(m, requires_grad=True)

# 	# solve the problem
# 	solution, = cvxpylayer(A_tch, b_tch)


# def _cvx_measure_mean_intensity(x,alpha=None,eps=1e-12):
# 	""" measure the batch-wise mean intensity of grayscale images """
# 	n=x.shape[0]
# 	if alpha is not None:
# 		m=torch.sum((x*alpha).reshape((n,-1)),dim=1)/(torch.sum(alpha.reshape((n,-1)),dim=1)+eps)
# 	else:
# 		m=torch.mean(x.reshape((n,-1)),dim=1)
# 	return m

# def _inverse_sigmoid(x,eps=1e-12):
# 	return torch.log((x+eps)/(1-x-eps))

# def _adjustment_operation(x,factor,bias):
# 	y=_inverse_sigmoid(x)
# 	y=y*factor+bias
# 	y=torch.sigmoid(y)


# def adjust_mean_var_intensity(x,alpha,target_mean,target_var):

# 	x
# 	n, m = 2, 3
# 	x = cp.Variable(n)
# 	A = cp.Parameter((m, n))
# 	b = cp.Parameter(m)
# 	constraints = [x >= 0]
# 	objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
# 	problem = cp.Problem(objective, constraints)
# 	assert problem.is_dpp()

# 	cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
# 	A_tch = torch.randn(m, n, requires_grad=True)
# 	b_tch = torch.randn(m, requires_grad=True)

# 	# solve the problem
# 	solution, = cvxpylayer(A_tch, b_tch)
