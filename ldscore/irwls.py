# coding=utf-8
'''
(c) 2015 Brendan Bulik-Sullivan and Hilary Finucane

Iterativey re-weighted least squares.
迭代重新加权最小二乘。
'''
from __future__ import division
import numpy as np
import jackknife as jk


class IRWLS(object):

    '''
    Iteratively re-weighted least squares (FLWS).

    Parameters
    ----------
    x : np.matrix with shape (n, p)
        Independent variable.
        自变量
    y : np.matrix with shape (n, 1)
        Dependent variable.
        因变量
    update_func : function
        Transforms output of np.linalg.lstsq to new weights.
        方法：将np.linalg.lstsq的输出转化为新的权重
    n_blocks : int
        Number of jackknife blocks (for estimating SE via block jackknife).
        jackknife块的数量（为了通过块的jackknife估算SE）
    w : np.matrix with shape (n, 1)
        Initial regression weights (default is the identity matrix). These should be on the
        inverse CVF scale.
        初始回归权重（默认是单位矩阵）这应该是和1/CVF同样的规模。
    slow : bool
        Use slow block jackknife? (Mostly for testing)
        是否使用慢的jackknife？通常是为了测试而用
    Attributes
    ----------
    est : np.matrix with shape (1, p)
        IRWLS estimate.
        迭代重新加权最小二乘估计
    jknife_est : np.matrix with shape (1, p)
        Jackknifed estimate.
        Jackknife估计
    jknife_var : np.matrix with shape (1, p)
        Variance of jackknifed estimate.
        Jackknife估计的方差
    jknife_se : np.matrix with shape (1, p)
        Standard error of jackknifed estimate, equal to sqrt(jknife_var).
        jackknife估计的标准差
    jknife_cov : np.matrix with shape (p, p)
        Covariance matrix of jackknifed estimate.
        Jackknife估计的协方差矩阵
    delete_values : np.matrix with shape (n_blocks, p)
        Jackknife delete values.
        jackknife删除的值
    Methods
    -------
    wls(x, y, w) :
        Weighted Least Squares.
        加权最小二乘
    _weight(x, w) :
        Weight x by w.
        为x赋予权重w
    '''

    def __init__(self, x, y, update_func, n_blocks, w=None, slow=False, separators=None):
        n, p = jk._check_shape(x, y)
        if w is None:
            w = np.ones_like(y)
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        jknife = self.irwls(
            x, y, update_func, n_blocks, w, slow=slow, separators=separators)
        self.est = jknife.est
        self.jknife_se = jknife.jknife_se
        self.jknife_est = jknife.jknife_est
        self.jknife_var = jknife.jknife_var
        self.jknife_cov = jknife.jknife_cov
        self.delete_values = jknife.delete_values
        self.separators = jknife.separators

    # 类方法
    @classmethod
    def irwls(cls, x, y, update_func, n_blocks, w, slow=False, separators=None):
        '''
        Iteratively re-weighted least squares (IRWLS).

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Independent variable.
        y : np.matrix with shape (n, 1)
            Dependent variable.
        update_func: function
            Transforms output of np.linalg.lstsq to new weights.
        n_blocks : int
            Number of jackknife blocks (for estimating SE via block jackknife).
        w : np.matrix with shape (n, 1)
            Initial regression weights.
        slow : bool
            Use slow block jackknife? (Mostly for testing)
        separators : list or None
            Block jackknife block boundaries (optional).

        Returns
        -------
        jknife : jk.LstsqJackknifeFast
            Block jackknife regression with the final IRWLS weights.
            用最终的权重进行的块jackknife回归
        '''
        (n, p) = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                'y has shape {S}. y must have shape ({N}, 1).'.format(S=y.shape, N=n))
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        # 以下是怎么回事
        w = np.sqrt(w)
        for i in xrange(2):  # update this later
            new_w = np.sqrt(update_func(cls.wls(x, y, w)))
            if new_w.shape != w.shape:
                print 'IRWLS update:', new_w.shape, w.shape
                raise ValueError('New weights must have same shape.')
            else:
                w = new_w

        x = cls._weight(x, w)
        y = cls._weight(y, w)

        # 如果是slow,怎么怎么样。将来可以去掉
        if slow:
            jknife = jk.LstsqJackknifeSlow(
                x, y, n_blocks, separators=separators)
        else:
            jknife = jk.LstsqJackknifeFast(
                x, y, n_blocks, separators=separators)

        return jknife

    @classmethod
    def wls(cls, x, y, w):
        '''
        Weighted least squares.
        加权最小二乘

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Independent variable.
        y : np.matrix with shape (n, 1)
            Dependent variable.
        w : np.matrix with shape (n, 1)
            Regression weights (1/CVF scale).

        Returns
        -------
        coef : list with four elements (coefficients, residuals, rank, singular values)
            Output of np.linalg.lstsq
            四个元素的列表（系数，残差，秩，奇异值）
        '''
        (n, p) = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                'y has shape {S}. y must have shape ({N}, 1).'.format(S=y.shape, N=n))
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape ({N}, 1).'.format(S=w.shape, N=n))

        # 给自变量和因变量赋权重
        x = cls._weight(x, w)
        y = cls._weight(y, w)
        # 通过计算使欧几里得2-范数最小化的向量“x”来求解方程“a x = b”。 b - a x || ^ 2`。
        # 该方程可以是低于，好的或高于确定的（即，a的线性独立行的数目可以小于，等于或大于线性独立列的数目）。
        # 如果“a”是正方形且满秩，则“x”（但是对于舍入误差）是方程的“精确”解。
        coef = np.linalg.lstsq(x, y)
        return coef

    @classmethod
    def _weight(cls, x, w):
        '''
        Weight x by w.

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Rows are observations.
        w : np.matrix with shape (n, 1)
            Regression weights (1 / sqrt(CVF) scale).
            什么是CVF ????

        Returns
        -------
        x_new : np.matrix with shape (n, p)
            x_new[i,j] = x[i,j] * w'[i], where w' is w normalized to have sum 1.
                                        w‘是w正规化的，和为1.
        Raises
        ------
        ValueError :
            If any element of w is <= 0 (negative weights are not meaningful in WLS).
            w中有负值就报错，因为在最小二乘法中负值是没有意义的
        '''
        if np.any(w <= 0):
            raise ValueError('Weights must be > 0')
        (n, p) = x.shape
        if w.shape != (n, 1):
            raise ValueError(
                'w has shape {S}. w must have shape (n, 1).'.format(S=w.shape))

        w = w / float(np.sum(w))
        x_new = np.multiply(x, w)
        return x_new
