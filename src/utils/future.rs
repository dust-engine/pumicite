use std::pin::Pin;

pub struct YieldNow(bool);
impl Future for YieldNow {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            std::task::Poll::Pending
        }
    }
}

pub fn yield_now() -> YieldNow {
    YieldNow(false)
}

pub fn zip_many<F: Future>(futures: impl Iterator<Item = F>) -> ZipMany<F> {
    ZipMany {
        futures: futures.map(|x| (x, None)).collect(),
    }
}

pub struct ZipMany<F: Future> {
    futures: Box<[(F, Option<F::Output>)]>,
}
impl<F: Future> Future for ZipMany<F> {
    type Output = Box<[F::Output]>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        let mut all_completed = true;
        for (f, out) in this.futures.iter_mut() {
            if out.is_some() {
                continue;
            }
            all_completed = false;
            let pinned = unsafe { Pin::new_unchecked(f) };
            match pinned.poll(cx) {
                std::task::Poll::Ready(result) => {
                    out.replace(result);
                }
                std::task::Poll::Pending => (),
            };
        }

        if all_completed {
            let outputs: Box<[F::Output]> = this
                .futures
                .iter_mut()
                .map(|x| x.1.take().unwrap())
                .collect();
            std::task::Poll::Ready(outputs)
        } else {
            std::task::Poll::Pending
        }
    }
}
