import { Link } from 'react-router';
import { Navigation } from '../components/Navigation';
import { STITCH_HERO_SCREENS } from '../../lib/lab-routes';
export function NotFound() {
  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-6 md:px-12 max-w-3xl mx-auto text-center">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-[#00dbe9] mb-2">404 // ROUTE_NOT_FOUND</p>
        <h1 className="text-4xl font-bold text-white mb-4">Unknown sector</h1>
        <p className="text-[#b9cacb] mb-10">
          This path is not mapped in the QuLab GUI. Use mission control or pick a hero screen below.
        </p>
        <div className="flex flex-wrap justify-center gap-4 mb-12">
          <Link
            to="/"
            className="px-6 py-3 bg-[#00dbe9] text-[#00363a] text-xs font-bold uppercase tracking-wider"
          >
            Mission control
          </Link>
          <Link
            to="/labs"
            className="px-6 py-3 border border-[#00dbe9]/40 text-[#00dbe9] text-xs font-bold uppercase tracking-wider"
          >
            All labs
          </Link>
          <Link
            to="/screens"
            className="px-6 py-3 border border-[#00dbe9]/40 text-[#00dbe9] text-xs font-bold uppercase tracking-wider"
          >
            Stitch screens
          </Link>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 text-left">
          {STITCH_HERO_SCREENS.map((screen) => (
            <Link
              key={screen.id}
              to={screen.path}
              className="glass-panel rounded-lg border border-white/10 p-4 hover:border-[#00dbe9]/50"
            >
              <span className="text-sm font-semibold text-white">{screen.title}</span>
              <p className="text-xs text-[#849495] mt-1">{screen.path}</p>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
