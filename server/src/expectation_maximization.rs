use ndarray::prelude::*;
use std::f64::consts::PI;
use std::f64::consts::E;

// E step of EM algorithm with missing data
// input: X: n*d data matrix;
//        Mu: K*d matrix, each row corresponds to a mixture mean;
//        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
// output: a Loglikelihood value
fn log_n(x: &Array1<f64>, mu: &Array1<f64>, var: f64) -> f64 {
    let shape = x.shape();
    let x_diff = &Array1::from(x-mu);
    let x_norm = x_diff.dot(x_diff);
    let e_exponent = -x_norm/(var* 2.);
    
    let theta = (PI*2.*var).ln();
    return e_exponent*E.ln() - (shape[0] as f64)/2.*theta;
}

#[test]
fn test_log_n() {
    let x = &array![1.,0.,0.];
    let mu = &array![0.,0.,0.];
    let var = 10.;

    let result = log_n(x, mu, var);
    assert_eq!(result, -6.260693239105087);
}

// E step of EM algorithm with missing data
// input: X: n*d data matrix;
//        K: number of mixtures;
//        Mu: K*d matrix, each row corresponds to a mixture mean;
//        P: K*1 matrix, each entry corresponds to the weight for a mixture;
//        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
// output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
//        LL: a Loglikelihood value
pub fn e_step(x: &Array2<f64>, mu: &Array2<f64>, p: &Array1<f64>, var: &Array1<f64>) -> (Array2<f64>, f64) {
    let x_shape = x.shape();
    let mu_shape = mu.shape();
    let mut post = Array2::<f64>::zeros((x_shape[0], mu_shape[0]).f());
    let mut ll = 0.;
    
    for (x_index,x_item) in x.outer_iter().enumerate() {
        let delta = x_item.iter()
            .map(|&e| if e != 0. {true} else {false})
            .collect::<Vec<bool>>();

        let existing_x = &Array1::from(
            x_item.iter()
                .enumerate()
                .filter(|(i, _)| delta[i.to_owned()] == true)
                .map(|(_, &e)| e)
                .collect::<Vec<f64>>()
        );

        let mut likelihoods = Vec::<f64>::new();
        
        for j in 0..mu_shape[0] {
            let existing_mu = &Array1::from(
                mu.index_axis(Axis(0), j).iter()
                .enumerate()
                .filter(|(i, _)| delta[i.to_owned()] == true)
                .map(|(_, &e)| e)
                .collect::<Vec<f64>>()
            );
            let log_scaled_weighted_density = p[j].ln() + log_n(existing_x, existing_mu, var[j]);
            likelihoods.push(log_scaled_weighted_density)
        }

        let x_prime = likelihoods.iter().cloned().fold(0./0., f64::max);
        let shifted_sum = likelihoods.iter().cloned().fold(0., |a, b| a + (b-x_prime).exp()); // logarithm magic
        let likelihoodsum = x_prime + shifted_sum.ln(); // more logarithm magic
        // # print np.log(max())[0]
        ll += likelihoodsum;
        println!(" {:?} {:?}", ll, likelihoodsum);

        for j in 0..mu_shape[0] {
            let existing_mu = &Array1::from(
                mu.index_axis(Axis(0), j).iter()
                .enumerate()
                .filter(|(i, _)| delta[i.to_owned()] == true)
                .map(|(_, &e)| e)
                .collect::<Vec<f64>>()
            );
            let log_scaled_weighted_density = p[j].ln() + log_n(existing_x, existing_mu, var[j]);
            post[[x_index, j]] = (log_scaled_weighted_density - likelihoodsum).exp();
        }
    }
    
    (post, ll)
}

#[test]
fn test_e_step() {
    let x = &array![[1.,0.,0.], [0.,1.,0.]];
    let mu = &array![[0.,1.,0.], [1.,0.,0.]];
    let p = &array![10., 1.];
    let var = &array![10., 1.];

    let result = e_step(x, mu, p, var);
    assert_eq!(result.0, array![[0.7505022115281781, 0.24949778847182194],
        [0.8390656652626518, 0.16093433473734806]]);
    assert_eq!(result.1, 0.8771870172298175);
}

// # M step of EM algorithm
// # input: X: n*d data matrix;
// #        K: number of mixtures;
// #        Mu: K*d matrix, each row corresponds to a mixture mean;
// #        P: K*1 matrix, each entry corresponds to the weight for a mixture;
// #        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
// #        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
// # output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
// #        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
// #        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
// def Mstep_part2(X,K,Mu,P,Var,post, minVariance=0.25):
pub fn m_step(x: &Array2<f64>, mut mu: Array2<f64>, mut p: Array1<f64>, mut var: Array1<f64>, post: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let x_shape = x.shape();
    let mu_shape = mu.shape();
    
    for j in 0..mu_shape[0] {
        let nj = post.slice(s![.., j]).sum();
        p[j] = nj/(x_shape[0] as f64);

        let mut newmux = Array1::<f64>::zeros(x_shape[1]);
        let mut newmutotal = Array1::<f64>::zeros(x_shape[1]);
        for (x_index,x_item) in x.outer_iter().enumerate() {
            let delta = x_item.mapv(|e| if e != 0. {1.} else {0.});
            newmux = newmux+delta.clone()*x_item*post[[x_index, j]];
            newmutotal = newmutotal+delta.clone()*post[[x_index, j]];
            // println!("{:?}", newmux);
        };
        
        for (x_index, total) in newmutotal.iter().enumerate() {
            if total >= &1. {
                mu[[j, x_index]] = newmux[x_index]/total/total;
            }
        }

        let mut newvar = 0.;
        let mut newvartotal = 0.;
        for (x_index, x_item) in x.outer_iter().enumerate() {
            let delta = x_item.mapv(|e| if e != 0. {1.} else {0.});
            
            let x_item = x_item.mapv(|a| a)*delta.clone();
            let existing_mu = delta.clone()*mu.slice(s![j, ..]);
            let x_diff = &Array1::from(x_item-existing_mu);
            let x_norm = x_diff.dot(x_diff);
            newvar += x_norm*post[[x_index, j]];

            let deltasize = delta.iter()
                .filter(|&&e| e != 0.).count();
            newvartotal = newvartotal+(deltasize as f64)*post[[x_index, j]];
        }
        // var[j] = max(newvar/newvartotal, minVariance)
        var[j] = newvar/newvartotal;
    }

    (mu, p, var)
}

#[test]
fn test_m_step() {
    let x = &array![[1.,0.,0.], [0.,1.,0.]];
    let mu = array![[0.,1.,0.], [1.,0.,0.]];
    let p = array![10., 1.];
    let var = array![10., 1.];
    let post = &array![[0.75050221, 0.24949779], [0.83906567, 0.16093433]];

    let result = m_step(x, mu, p, var, post);
    assert_eq!(result.0, array![[0., 1., 0.],
            [1., 0., 0.]]);
            
    assert_eq!(result.1, array![0.79478394, 0.20521605999999998]);
    assert_eq!(result.2, array![0.4721422843546637, 0.3921094918204745]);
}