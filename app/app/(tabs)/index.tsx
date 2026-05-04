import { Redirect } from 'expo-router';
import { useMemo } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AllocationBar } from '@/components/AllocationBar';
import { DisclaimerBanner } from '@/components/DisclaimerBanner';
import { MetricCard, MetricGrid } from '@/components/MetricCard';
import { PortfolioChart } from '@/components/PortfolioChart';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { loadProfileDetail, summary } from '@/data/loadData';
import { formatCurrency, formatPercent, scaleFromBase } from '@/lib/format';
import { ProfileColors, SemanticColors } from '@/lib/palette';
import { useAppState } from '@/state/AppContext';

export default function DashboardScreen() {
  const { ready, onboarded, choice } = useAppState();

  const detail = useMemo(
    () => loadProfileDetail(choice.profileKey, choice.tiltEnabled),
    [choice.profileKey, choice.tiltEnabled]
  );

  if (!ready) return <ThemedView style={styles.flex} />;
  if (!onboarded) return <Redirect href="/onboarding" />;
  const profileColor = ProfileColors[choice.profileKey].primary;

  const finalValueScaled = scaleFromBase(detail.metrics.final_portfolio_value, choice.capital);
  const returnDollars = finalValueScaled - choice.capital;
  const returnColor =
    returnDollars >= 0 ? SemanticColors.positive : SemanticColors.negative;

  const benchmarkFinal = scaleFromBase(
    summary.benchmark.metrics.final_portfolio_value,
    choice.capital
  );
  const beatBenchmark = finalValueScaled - benchmarkFinal;

  const lastDay = detail.daily[detail.daily.length - 1];

  return (
    <ThemedView style={styles.root}>
      <SafeAreaView style={styles.flex} edges={['top']}>
        <ScrollView contentContainerStyle={styles.scroll}>
          <View style={styles.headerRow}>
            <View style={styles.headerLeft}>
              <ThemedText style={styles.eyebrow}>{detail.profile.name} profile</ThemedText>
              <ThemedText type="title" style={styles.value}>
                {formatCurrency(finalValueScaled)}
              </ThemedText>
              <ThemedText style={[styles.delta, { color: returnColor }]}>
                {returnDollars >= 0 ? '+' : ''}
                {formatCurrency(returnDollars)} ({formatPercent(detail.metrics.total_return)})
              </ThemedText>
            </View>
            <View style={[styles.tiltBadge, { borderColor: profileColor }]}>
              <ThemedText style={[styles.tiltLabel, { color: profileColor }]}>
                {choice.tiltEnabled ? 'AI tilt on' : 'no AI tilt'}
              </ThemedText>
            </View>
          </View>

          <ThemedText style={styles.helper}>
            Simulated outcome of investing {formatCurrency(choice.capital)} on{' '}
            {summary.simulation.start_date} and rebalancing quarterly through{' '}
            {summary.simulation.end_date}.
          </ThemedText>

          <DisclaimerBanner text={summary.disclaimer} compact />

          <View style={styles.section}>
            <ThemedText type="subtitle">Portfolio value over time</ThemedText>
            <PortfolioChart
              daily={detail.daily}
              benchmark={summary.benchmark.daily}
              capital={choice.capital}
              primaryColor={profileColor}
            />
            <View style={styles.benchmarkCallout}>
              <ThemedText style={styles.benchmarkText}>
                vs SPY buy-and-hold ({formatCurrency(benchmarkFinal)},{' '}
                {formatPercent(summary.benchmark.metrics.total_return)}):
                <ThemedText
                  style={[styles.benchmarkText, { color: beatBenchmark >= 0 ? SemanticColors.positive : SemanticColors.negative }]}
                >
                  {' '}
                  {beatBenchmark >= 0 ? '+' : ''}
                  {formatCurrency(beatBenchmark)}
                </ThemedText>
              </ThemedText>
            </View>
          </View>

          <View style={styles.section}>
            <ThemedText type="subtitle">Key metrics</ThemedText>
            <MetricGrid>
              <MetricCard
                label="Annualized return"
                value={formatPercent(detail.metrics.annualized_return)}
                valueColor={returnColor}
              />
              <MetricCard
                label="Max drawdown"
                value={formatPercent(detail.metrics.max_drawdown)}
                caption={detail.metrics.max_drawdown_date ?? undefined}
                valueColor={SemanticColors.negative}
              />
              <MetricCard
                label="Sharpe ratio"
                value={
                  detail.metrics.sharpe_ratio != null
                    ? detail.metrics.sharpe_ratio.toFixed(2)
                    : '—'
                }
                caption=">1 is decent, >2 is excellent"
              />
              <MetricCard
                label="Win rate"
                value={formatPercent(detail.metrics.win_rate, 1)}
                caption="% positive trading days"
              />
              <MetricCard
                label="Trades placed"
                value={String(detail.metrics.total_trades)}
                caption={`$${detail.metrics.total_transaction_costs.toFixed(2)} fees (sim)`}
              />
              <MetricCard
                label="Quarters simulated"
                value={String(detail.rebalance_events.length)}
                caption="rebalances"
              />
            </MetricGrid>
          </View>

          <View style={styles.section}>
            <ThemedText type="subtitle">Current allocation</ThemedText>
            <ThemedText style={styles.helper}>
              As of the last simulated trading day. Drift between rebalances is normal —
              equity tends to grow above target in bull markets.
            </ThemedText>
            {lastDay ? (
              <AllocationBar
                weights={lastDay.asset_class_weights}
                labels={summary.asset_class_labels}
              />
            ) : null}
          </View>
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },
  scroll: { padding: 16, gap: 20, paddingBottom: 40 },
  headerRow: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' },
  headerLeft: { flex: 1, gap: 4 },
  eyebrow: { fontSize: 12, opacity: 0.7, textTransform: 'uppercase', letterSpacing: 0.5 },
  value: { fontSize: 32, fontWeight: '700' },
  delta: { fontSize: 14, fontWeight: '600' },
  tiltBadge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  tiltLabel: { fontSize: 11, fontWeight: '600' },
  helper: { fontSize: 12, opacity: 0.7, lineHeight: 18 },
  section: { gap: 10 },
  benchmarkCallout: {
    paddingVertical: 4,
  },
  benchmarkText: { fontSize: 13, opacity: 0.9 },
});
