library(tidyverse)
library(latex2exp)
library(ppcor)
library(furrr)

plan(multisession, workers = 3)

test_pcor_fast <- function(N,M,sigma_theta=1,sigma_phi=1,
                           sigma_u=1,sigma_t=1,sigma_y=1,gamma=1,lambda=gamma,beta=1) {

  theta <- rnorm(M)*sigma_theta
  phi <- rnorm(M)*sigma_phi
  psi <- 0
  U1 <- phi + rnorm(M)*sigma_u
  T1 <- gamma*U1 + theta + rnorm(M)*sigma_t
  Y1  <- lambda*U1 + beta*T1 + psi + rnorm(M)*sigma_y
  U2 <- phi + rnorm(M)*sigma_u
  T2 <- gamma*U2 + theta + rnorm(M)*sigma_t

  pcor.test(Y1,T2,T1)
}

test_setting_fast <- function(M=1000,sigma_theta=1,sigma_phi=1,sigma_u=1,sigma_t=1,gamma=gamma,lambda=lambda) {
  mean(map_dbl(1:1000,~{test_pcor_fast(2,M,sigma_theta=sigma_theta,sigma_phi=sigma_phi,sigma_u=sigma_u,sigma_t=sigma_t,gamma=gamma,lambda=lambda)[[2]]})<0.05)
}

system.time(
  df_res_gamma <- expand_grid(gamma=seq(0.001,15,length.out=40),
                          M =seq(500,4000,by=100)) %>%
    mutate(rate=future_map2_dbl(gamma,M,~test_setting_fast(M=.y,sigma_u=1,sigma_t=2/3,gamma=.x,lambda=.x),.options=furrr_options(seed=TRUE)))
)

system.time(
  df_res_lambda <- expand_grid(gamma=seq(0.001,15,length.out=40),
                                  M =seq(500,4000,by=100)) %>%
    mutate(rate=future_map2_dbl(gamma,M,~test_setting_fast(M=.y,sigma_u=1,sigma_t=2/3,gamma=1,lambda=.x),.options=furrr_options(seed=TRUE)))
)

df_res_gamma %>%
  ggplot(aes(x=M, y=gamma, z=rate)) +
  geom_contour_filled() +
  #geom_tile() +
  #scale_x_continuous(trans = "log2") +
  #scale_y_continuous(trans = "log2") +
  #coord_equal() +
  scale_fill_viridis_d(option="B") +
  labs(x=TeX("$M$"),
       y=TeX("$\\gamma$"),
       fill=TeX("$p_{detection}$")) +
  theme_minimal() +
  geom_abline(slope=3/2,intercept=0,color="white") +
  geom_hline(yintercept=1)
ggsave("example1_continuous_gamma.pdf",width=3,height=4.5)
write_csv(df_res_gamma,file = "example1_continuous_gamma.csv")

df_res_lambda %>%
  ggplot(aes(x=M, y=gamma, z=rate)) +
  geom_contour_filled() +
  #geom_tile() +
  scale_fill_viridis_d(option="B") +
  labs(x=TeX("$M$"),
       y=TeX("$\\lambda$"),
       fill=TeX("$p_{detection}$")) +
  theme_minimal() +
  geom_abline(slope=3/2,intercept=0,color="white")
ggsave("example1_continuous_lambda.pdf",width=3,height=4.5)
write_csv(df_res_lambda,file = "example1_continuous_lambda.csv")

