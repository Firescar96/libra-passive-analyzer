use ndarray::prelude::*;
use ndarray_linalg::norm::*;
use std::f64::consts::PI;
use std::f64::consts::E;

// def log_N(x,mu,var):
// (d,) = np.shape(x)
// d = float(d)
// squared_diff = np.power(LA.norm(x-mu),2)
// e_exponent = -squared_diff/(2*var)
// result = e_exponent*np.log(np.e) - d/2*np.log(np.pi*2*var)
// return result

// E step of EM algorithm with missing data
// input: X: n*d data matrix;
//        Mu: K*d matrix, each row corresponds to a mixture mean;
//        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
// output: a Loglikelihood value
fn log_n(x: &Array2<f64>, mu: &Array2<f64>, var: &Array1<f64>) -> Array1<f64> {
    let shape = x.shape();
    let (_, x_diff_vec) = normalize(x-mu, NormalizeAxis::Column);
    let x_diff = &Array1::from(x_diff_vec);
    let x_norm = x_diff.dot(x_diff);
    let e_exponent = -x_norm/(var* 2.);
    println!("{}", e_exponent);
    let theta = (PI*2.*var).iter().map(|v| v.ln()).collect::<Vec<f64>>();
    
    return e_exponent*E.ln() - (shape[0] as f64)/2.*Array1::from(theta);
}

#[test]
fn test_log_n() {
    let x = &array![[1.,0.,0.], [0.,1.,0.]];
    let mu = &array![[0.,0.,0.], [1.,0.,0.]];
    let var = &array![10., 1.];

    let result = log_n(x, mu, var);
    assert_eq!(result[0], -4.290462159403392);
    assert_eq!(result[1], -3.3378770664093453);
}

// def Estep_part2(X,K,Mu,P,Var):
//     n,d = np.shape(X) # n data points of dimension d
//     post = np.zeros((n,K)) # posterior probabilities to compute
//     LL = 0.0    # the LogLikelihood

//     for (t,x) in enumerate(X):
//         delta = np.zeros(d)
//         delta[x != 0] = 1
//         x = np.array([x[e] for e in range(d) if delta[e] == 1])

//         likelihoods = []
//         for j in range(K):
//             mu = np.array([Mu[j][e] for e in range(d) if delta[e] == 1])
//             logScaledWeightedDensity = np.add(np.log(P[j]), log_N(x,mu,Var[j]))
//             likelihoods.append(logScaledWeightedDensity)
//         x_prime = max(likelihoods)
//         shifted_sum = sum(map(lambda x: np.exp(x-x_prime), likelihoods)) #logarithm magic
//         likelihoodsum = x_prime + np.log(shifted_sum)#more logarithm magic
//         # print np.log(max())[0]
//         LL += likelihoodsum

//         for j in range(K):
//             mu = np.array([Mu[j][e] for e in range(d) if delta[e] == 1])
//             logScaledWeightedDensity = np.add(np.log(P[j]), log_N(x,mu,Var[j]))
//             post[t][j] = logScaledWeightedDensity - likelihoodsum


//     return (np.exp(post),LL)

// E step of EM algorithm with missing data
// input: X: n*d data matrix;
//        K: number of mixtures;
//        Mu: K*d matrix, each row corresponds to a mixture mean;
//        P: K*1 matrix, each entry corresponds to the weight for a mixture;
//        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
// output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
//        LL: a Loglikelihood value
pub fn e_step(a: i32) -> i32 {
    a*2
}

#[test]
fn it_adds_two() {
    assert_eq!(4, e_step(2));
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
//     n,d = np.shape(X) # n data points of dimension d

//     post = post.transpose()
//     for j in range(K):
//         nj = sum(post[j])
//         P[j] = nj/len(X)

//         newmux = 0
//         newmutotal  = 0
//         for (t,x) in enumerate(X):
//             delta = np.array([1 if x_prime != 0 else 0 for x_prime in x])
//             newmux += delta*x*post[j][t]
//             newmutotal += delta*post[j][t]
//         for (t, total) in enumerate(newmutotal):
//             if total >= 1:
//                 Mu[j][t] = newmux[t]/total

//         newvar = 0
//         newvartotal = 0
//         for (t,x) in enumerate(X):
//             delta = np.array([1 if x_prime != 0 else 0 for x_prime in x])
//             x = delta*x
//             mu = delta*Mu[j]
//             squared_diff = np.power(LA.norm(x-mu),2)
//             newvar += squared_diff*post[j][t]

//             deltasize = len(filter(lambda x: x != 0, delta))
//             newvartotal += deltasize*post[j][t]
//         Var[j] = max(newvar/newvartotal, minVariance)

//     return (Mu,P,Var)