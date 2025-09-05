use std::env;
use std::path::{PathBuf};
use csv::Reader;
use serde::Deserialize;
use polars;

use std::iter::zip;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Deserialize)]
pub struct Lightcurve {
    t: Array1<f64>,
    val: Array1<f64>,
    err: Array1<f64>,
    redshift: f64
}


pub fn sfdata(
    t: Array1<f64>,
    val: Array1<f64>,
    err: Array1<f64>,
    redshift: f64
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let len_ts = t.len();

    if len_ts < 2 {
        return (array![], array![], array![])
    }

    let len_tau = (len_ts * (len_ts - 1)) >> 1;

    let mut tau: Array1<f64> = Array1::<f64>::default(len_tau);
    let mut delta: Array1<f64> = Array1::<f64>::default(len_tau);
    let mut sigma: Array1<f64> = Array1::<f64>::default(len_tau);

    for i in 0..len_ts-1 {
        let span = len_ts - 1 - i;
        let begin = i*len_ts - ( (i*(i+1))>>1 );
        let end = begin + span;

        let dt = &t.slice(s![i+1..]) - t[i];
        tau.slice_mut(s![begin..end]).assign(&dt);

        let dval = &val.slice(s![i+1..]) - val[i];
        delta.slice_mut(s![begin..end]).assign(&dval);

        let dval_err = (&err.slice(s![i+1..]).powi(2) + err[i].powi(2)).sqrt();
        sigma.slice_mut(s![begin..end]).assign(&dval_err);
    }

    (tau/(1.0+redshift), delta, sigma)
}


pub fn esfdata(lcurves: Vec<Lightcurve>)
    -> (Array1<f64>, Array1<f64>, Array1<f64>) {

    let size_ts: Vec<usize> = lcurves.iter().map(|x| x.t.len()).collect();
    let size_tau: Vec<usize> = size_ts.iter().map(|x| x*(x-1)>>1).collect();

    let len_result: usize = (&size_tau).into_iter().sum();

    let mut tau: Array1<f64> = Array1::<f64>::default(len_result);
    let mut delta: Array1<f64> = Array1::<f64>::default(len_result);
    let mut sigma: Array1<f64> = Array1::<f64>::default(len_result);

    let mut begin: usize = 0;
    for (ele_size_tau, lcurve) in zip(size_tau, lcurves) {
        let end = begin + ele_size_tau;

        let (_tau, _delta, _sigma) = sfdata(lcurve.t, lcurve.val, lcurve.err, lcurve.redshift);
        tau.slice_mut(s![begin..end]).assign(&_tau);
        delta.slice_mut(s![begin..end]).assign(&_delta);
        sigma.slice_mut(s![begin..end]).assign(&_sigma);

        begin = end;
    }

    (tau, delta, sigma)
}


pub fn esfdata_parallel(lcurves: Vec<Lightcurve>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // 计算每个光变曲线的数据点数量
    let size_ts: Vec<usize> = lcurves.iter().map(|x| x.t.len()).collect();

    // 计算每个光变曲线需要的结果数组大小
    let size_tau: Vec<usize> = size_ts.iter()
        .map(|&x| if x < 2 { 0 } else { x * (x - 1) / 2 })
        .collect();

    // 计算总结果数组大小
    let len_result: usize = size_tau.iter().sum();

    // 预分配结果数组
    let tau: Array1<f64> = Array1::<f64>::default(len_result);
    let delta: Array1<f64> = Array1::<f64>::default(len_result);
    let sigma: Array1<f64> = Array1::<f64>::default(len_result);

    // 使用原子变量跟踪当前写入位置
    let current_offset = AtomicUsize::new(0);

    // 使用 Mutex 包装数组以安全地在多线程中访问
    let tau_mutex = Mutex::new(tau);
    let delta_mutex = Mutex::new(delta);
    let sigma_mutex = Mutex::new(sigma);

    // 使用 Rayon 并行处理每个光变曲线
    lcurves.into_par_iter().for_each(|lcurve| {
        // 跳过点数不足的光变曲线
        if lcurve.t.len() < 2 {
            return;
        }

        // 调用 sfdata 处理当前光变曲线
        let (_tau, _delta, _sigma) = sfdata(
            lcurve.t,
            lcurve.val,
            lcurve.err,
            lcurve.redshift
        );

        // 获取当前偏移量并更新
        let offset = current_offset.fetch_add(_tau.len(), Ordering::SeqCst);

        // 锁定数组并写入结果
        {
            let mut tau_lock = tau_mutex.lock().unwrap();
            tau_lock.slice_mut(s![offset..offset + _tau.len()]).assign(&_tau);
        }

        {
            let mut delta_lock = delta_mutex.lock().unwrap();
            delta_lock.slice_mut(s![offset..offset + _delta.len()]).assign(&_delta);
        }

        {
            let mut sigma_lock = sigma_mutex.lock().unwrap();
            sigma_lock.slice_mut(s![offset..offset + _sigma.len()]).assign(&_sigma);
        }
    });

    // 获取最终结果
    let tau = tau_mutex.into_inner().unwrap();
    let delta = delta_mutex.into_inner().unwrap();
    let sigma = sigma_mutex.into_inner().unwrap();

    (tau, delta, sigma)
}


pub fn calc_esf_task() {

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let t = array![20., 30., 50., 90., 100.];
        let val = array![3., 1., 9., 4., 6.];
        let err = array![1., 3., 1., 2., 1.];

        let (tau, dval, _) = sfdata(t, val, err, 0.);

        assert_eq!(tau, array![10., 30., 70., 80., 20., 60., 70., 40., 50., 10.]);
        assert_eq!(dval, array![-2., 6., 1., 3., 8., 3., 5., -5., -3., 2.]);
        // assert_eq!(sig.powi(2), array![10., 2., 5., 2., 10., 13., 10., 5., 2., 5.]);
    }

    #[test]
    fn test_esfdata() {
        let lc_vec = vec![
            Lightcurve {
                t: array![20., 30., 50., 90., 100.],
                val: array![3., 1., 9., 4., 6.],
                err: array![1., 3., 1., 2., 1.],
                redshift: 0.0
            }
        ];

        esfdata(lc_vec);
    }

    #[test]
    fn test_esfdata_parallel() {
        let lc_vec = vec![
            Lightcurve {
                t: array![20., 30., 50., 90., 100.],
                val: array![3., 1., 9., 4., 6.],
                err: array![1., 3., 1., 2., 1.],
                redshift: 0.0
            },
            Lightcurve {
                t: array![1., 2., 3.],
                val: array![0.1, 0.2, 0.3],
                err: array![0.01, 0.02, 0.03],
                redshift: 0.5
            }
        ];

        let (tau, delta, sigma) = esfdata(lc_vec);

        // 验证结果数组大小
        assert_eq!(tau.len(), 10 + 3); // 第一个曲线10个点，第二个3个点
        assert_eq!(delta.len(), 10 + 3);
        assert_eq!(sigma.len(), 10 + 3);

        // 验证部分数据点
        assert!((tau[0] - 10.0).abs() < 1e-6);
        assert!((delta[0] - (-2.0)).abs() < 1e-6);
    }
}
