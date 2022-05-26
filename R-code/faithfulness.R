library(tidyverse)
library(latex2exp)
library(ppcor)
library(furrr)

plan(multisession, workers = 3)

test_pcor_fast <- function(N,M,sigma_theta=1,sigma_phi=1,
                           sigma_u=1,sigma_t=1,sigma_y=1) {

  theta <- rnorm(M)*sigma_theta
  phi <- rnorm(M)*sigma_phi
  psi <- 0
  U1 <- phi + rnorm(M)*sigma_u
  T1 <- U1 + theta + rnorm(M)*sigma_t
  Y1  <- U1 + psi + rnorm(M)*sigma_y
  U2 <- phi + rnorm(M)*sigma_u
  T2 <- U2 + theta + rnorm(M)*sigma_t

  pcor.test(Y1,T2,T1)
}

test_setting_fast <- function(sigma_theta,sigma_phi,sigma_u,sigma_t=1) {
  mean(map_dbl(1:1000,~{test_pcor_fast(2,1000,sigma_theta=sigma_theta,sigma_phi=sigma_phi,sigma_u=sigma_u,sigma_t=sigma_t)[[2]]})<0.05)
}

system.time(
  df_res_1 <- expand_grid(sigma_theta=seq(0.001,5,length.out=100),
                          sigma_phi  =seq(0.001,5,length.out=100)) %>%
    mutate(rate=future_map2_dbl(sigma_theta,sigma_phi,~test_setting_fast(.x,.y,sigma_u=1,sigma_t=2/3),.options=furrr_options(seed=TRUE)))
)

df_res_1 %>%
  ggplot(aes(x=sigma_theta,y=sigma_phi, z=rate)) +
  geom_contour_filled() +
  #geom_tile() +
  #scale_x_continuous(trans = "log2") +
  #scale_y_continuous(trans = "log2") +
  coord_equal() +
  scale_fill_viridis_d(option="B") +
  labs(title=TeX("$\\sigma_U=1$, $\\sigma_T=\\frac{2}{3}$, $N=2$, $M=1000$"),
       x=TeX("$\\sigma_{\\theta_T}$"),
       y=TeX("$\\sigma_{\\theta_U}$"),
       fill=TeX("$p_{detection}$")) +
  theme_minimal() +
  geom_abline(slope=3/2,intercept=0,color="white")

write_csv(df_res_1,file = "faithfulness_k1000_r1000.csv")
ggsave("faithfulness_k1000_r1000.pdf",width=3,height=4.5)
