// Static data loader. Metro bundles the JSON files at build time, so all
// imports are resolved immediately — no async/network. To refresh the data,
// re-run `python main.py` in risk-return-analysis/ and rebuild the app.

import type { ProfileDetail, ProfileKey, Summary } from './types';

import summaryJson from '@/assets/data/summary.json';
import aggressiveTilt from '@/assets/data/aggressive_tilt.json';
import aggressiveNoTilt from '@/assets/data/aggressive_no_tilt.json';
import balancedTilt from '@/assets/data/balanced_tilt.json';
import balancedNoTilt from '@/assets/data/balanced_no_tilt.json';
import conservativeTilt from '@/assets/data/conservative_tilt.json';
import conservativeNoTilt from '@/assets/data/conservative_no_tilt.json';

const DETAIL_BY_KEY: Record<string, unknown> = {
  conservative_no_tilt: conservativeNoTilt,
  conservative_tilt: conservativeTilt,
  balanced_no_tilt: balancedNoTilt,
  balanced_tilt: balancedTilt,
  aggressive_no_tilt: aggressiveNoTilt,
  aggressive_tilt: aggressiveTilt,
};

export const summary = summaryJson as unknown as Summary;

export function loadProfileDetail(profileKey: ProfileKey, useTilt: boolean): ProfileDetail {
  const key = `${profileKey}_${useTilt ? 'tilt' : 'no_tilt'}`;
  const detail = DETAIL_BY_KEY[key];
  if (!detail) {
    throw new Error(`No bundled data for profile combo: ${key}`);
  }
  return detail as ProfileDetail;
}

export function findEvent(detail: ProfileDetail, isoDate: string) {
  return detail.rebalance_events.find((e) => e.date === isoDate);
}

export function findDaily(detail: ProfileDetail, isoDate: string) {
  return detail.daily.find((d) => d.date === isoDate);
}

export function previousEvent(detail: ProfileDetail, isoDate: string) {
  const events = detail.rebalance_events;
  let prev: typeof events[number] | undefined;
  for (const e of events) {
    if (e.date < isoDate) prev = e;
    else break;
  }
  return prev;
}
