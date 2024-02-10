from scipy.sparse.linalg import eigsh as ssl_eig
from tqdm import tqdm
import logging
import numpy as np
import torch
import pandas as pd


class DiffusionMaps:
    def __init__(
        self,
        data,
        batch_size=20,
        n_largest=100,
        abs_value=False,
        demean=True,
        how="pearson"
    ):
        self.data = torch.Tensor(data)
        self.batch_size = batch_size
        self.n_largest = n_largest
        self.abs_value = abs_value
        self.demean = demean
        self.how = how

    def estimate_memory_usage(self, tensor, datatype_size=4):
        """
        Estimate the memory usage of a tensor in bytes.

        Args:
        tensor (torch.Tensor): The tensor to estimate memory for.
        datatype_size (int): Size in bytes of the tensor's datatype (default is 4 for float32).

        Returns:
        int: Estimated memory usage in bytes.
        """
        return tensor.numel() * datatype_size

    def batch_cov_mem_cost(self, n, m, dtype=torch.float32):
        """
        Estimate the memory cost of computing the covariance between a batch of vectors and a whole batch of vectors.

        Args:
        n (int): The number of vectors in the batch.
        m (int): The number of vectors in the whole batch.
        dtype (torch.dtype): The datatype of the vectors.

        Returns:
        int: Estimated memory usage in bytes.
        """

        return (
            self.estimate_memory_usage(torch.empty(n, dtype=dtype))
            + self.estimate_memory_usage(torch.empty(m, dtype=dtype))
            + self.estimate_memory_usage(torch.empty(n, m, dtype=dtype))
        )

    def correlation(self, data=None, print_log=True):
        if data is None:
            data = self.data
        if self.how == "spearman":
            data = torch.argsort(torch.argsort(data, dim=0), dim=0).to(
                dtype=torch.float64
            )
        corr_collections = []
        batch_indices = np.array_split(
            np.arange(data.shape[0]), np.ceil(data.shape[0] / self.batch_size)
        )
        data_batches = torch.split_with_sizes(data, [len(x) for x in batch_indices])
        stds = data.std(dim=1).numpy()
        if print_log:
            logging.info("Computing correlation")
            iterable = tqdm(zip(batch_indices, data_batches), total=len(batch_indices))
        else:
            iterable = zip(batch_indices, data_batches)
        for batch_ind, batch in iterable:
            batch_cov = self.batch_corr(
                batch,
                data,
                stds,
                stds[batch_ind],
                n_largest=self.n_largest,
                abs_value=self.abs_value,
                demean=self.demean,
            )
            batch_corr_ = (
                pd.DataFrame(batch_cov)
                .assign(
                    source=np.concatenate(
                        [[ind for _ in range(self.n_largest)] for ind in batch_ind]
                    )
                )
                .set_index("source", append=True)
            )
            corr_collections.append(batch_corr_.stack())
        corr_collections = pd.concat(corr_collections)
        if self.how == "spearman":
            corr_collections = corr_collections.add_prefix("spearman_")
        return corr_collections

    def diffmaps(self, corr_df, n_eigs=20):
        sums = corr_df.groupby("source").sum() - 1
        corr_df = -corr_df / sums
        corr_df[
            (
                corr_df.index.get_level_values("source")
                == corr_df.index.get_level_values("target")
            )
        ] = 1
        lap = np.nan_to_num(corr_df.unstack().values)
        vals, vecs = ssl_eig(lap, k=n_eigs, which="SM", return_eigenvectors=True)
        return vals.real, vecs.real

    def batch_corr(
        self,
        part,
        whole,
        stds=None,
        sample_stds=None,
        n_largest=None,
        abs_value=False,
        demean=True,
    ):
        part = torch.Tensor(part)
        part_nanmean = torch.nanmean(part, dim=0, keepdim=True)
        whole_nanmean = torch.nanmean(whole, dim=0, keepdim=True)

        part = torch.where(torch.isnan(part), part_nanmean, part)
        whole = torch.where(torch.isnan(whole), whole_nanmean, whole)

        if demean:
            part = part - part.mean(dim=1, keepdim=True)
            whole = whole - whole.mean(dim=1, keepdim=True)

        cov = torch.matmul(part, whole.t())
        corr = cov / stds / sample_stds[:, None] / (whole.shape[1] - 1)

        corr_abs = torch.abs(corr) if abs_value else corr

        if n_largest is None:
            n_largest = corr_abs.shape[1]

        _corr, _ind = torch.topk(corr_abs, n_largest, dim=1)
        return pd.Series(
            _corr.flatten().numpy(),
            index=pd.Index(_ind.flatten().numpy(), name="target"),
            name="corr",
        )

    def fit(self, data=None, n_eigs=20):
        if data is None:
            data = self.data
        corr = self.correlation(data).unstack()
        eigs, vecs = self.diffmaps(corr, n_eigs=n_eigs)
        self.eigs = eigs
        self.vecs = vecs.T
        return eigs, vecs.T

    def predict(self, oos_data):
        corrs = self.batch_corr(
            oos_data,
            self.data,
            stds=self.data.std(dim=1).numpy(),
            sample_stds=oos_data.std(dim=1).numpy(),
            n_largest=self.n_largest,
            abs_value=self.abs_value,
            demean=self.demean,
        )
        corrs = (
            pd.DataFrame(corrs)
            .assign(
                source=np.concatenate(
                    [
                        [ind for _ in range(self.n_largest)]
                        for ind in range(oos_data.shape[0])
                    ]
                )
            )
            .set_index("source", append=True)
        )
        return corrs.groupby("source").apply(
            lambda df: pd.Series(
                self.vecs.T[df.index.get_level_values("target").values].mean(axis=0)
            )
        )


if __name__ == "__main__":
    # Load data
    data = torch.randn(10000, 200, dtype=torch.float64)
    oos_data = torch.randn(100, 200, dtype=torch.float64)

    diffmaps = DiffusionMaps(
        data, batch_size=100, n_largest=10, abs_value=False, demean=True, how="pearson"
    )
    # corr = diffmaps.correlation(print_log=True).unstack()
    diffmaps.fit(data, n_eigs=20)
    corrds = diffmaps.predict(oos_data)
