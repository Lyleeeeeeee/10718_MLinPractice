"""Pure original12 features; all input accesses stop at the declared cutoff."""
import calendar
import datetime as dt
import math
import statistics

def days(month):
    return calendar.monthrange(*map(int,month.split('-')))[1]


def shift(month,offset):
    year,m = map(int,month.split('-'))
    i = year*12+m-1+offset
    return f'{i//12:04d}-{i%12+1:02d}'


def historical_bound(rates,next_days):
    if len(rates)!=6 or any(v is None or not math.isfinite(v) or v<0 for v in rates):
        raise ValueError('Six published daily rates are required')
    median = statistics.median(rates)
    sigma = 1.4826*statistics.median(abs(v-median) for v in rates)
    return max(200,next_days*median+max(100,.5*next_days*median,3*next_days*sigma))


FEATURES = ['log_current_daily_rate','log_previous_daily_rate','log_mean_last3_daily_rate',
            'log_median_last6_daily_rate','last6_daily_rate_slope','log_seasonal_daily_rate',
            'seasonal_missing','log_current_month_mean_daily_PV','last14_prior14_log_PV_growth',
            'PV_month_missing','log_robust_daily_rate_sigma','log_daily_persistence_margin_U']


def article_features(article,cutoff,volumes,pageviews):
    target=shift(cutoff,1);history=[shift(cutoff,k) for k in range(-5,1)]
    rates=[volumes[m][article]/days(m) for m in history]
    median=statistics.median(rates);sigma=1.4826*statistics.median(abs(v-median) for v in rates)
    bound=historical_bound(rates,days(target));season=shift(target,-12)
    missing=article not in volumes.get(season,{})
    seasonal=rates[-1] if missing else volumes[season][article]/days(season)
    daily=pageviews.get(article,{})
    dates=[cutoff+f'-{i:02d}' for i in range(1,days(cutoff)+1)]
    complete=all(d in daily for d in dates)
    trend=None
    if complete:
        end=dt.date.fromisoformat(dates[-1])
        recent=[str(end-dt.timedelta(days=i)) for i in range(14)]
        previous=[str(end-dt.timedelta(days=i)) for i in range(14,28)]
        trend=math.log1p(sum(daily[d] for d in recent)/14)-math.log1p(sum(daily[d] for d in previous)/14)
    forecast=rates[-1]*days(target)
    features=[math.log1p(rates[-1]),math.log1p(rates[-2]),math.log1p(sum(rates[-3:])/3),
              math.log1p(median),sum((i-2.5)*r for i,r in enumerate(rates))/17.5,math.log1p(seasonal),int(missing),
              math.log1p(sum(daily[d] for d in dates)/len(dates)) if complete else None,trend,int(not complete),
              math.log1p(sigma),math.log1p(forecast)-math.log1p(bound)]
    return features,bound



def label_bound(rates, target_days):
    """Loose outcome bound only; never use this for candidate filtering."""
    if len(rates) != 6 or any(not math.isfinite(x) or x < 0 for x in rates):
        raise ValueError("Six nonnegative finite rates required")
    median = statistics.median(rates)
    sigma = 1.4826 * statistics.median(abs(x - median) for x in rates)
    return max(200., target_days * median + max(100., .25 * target_days * median, 2 * target_days * sigma))
