import { Link } from 'react-router';
import { Navigation } from '../components/Navigation';
import { STITCH_HERO_SCREENS } from '../../lib/lab-routes';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

/** Hub for Stitch hero screens — matches published figma.site flow after boot. */
export function StitchHub() {
  return (
    <div className="min-h-screen qulab-page-bg text-foreground">
      <Navigation />
      <main className="relative pt-32 pb-20 px-6 md:px-12 max-w-6xl mx-auto">
          <EchoLabCommandInline className="mb-8" />
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-[#00dbe9] mb-2">QuLab Infinite GUI</p>
        <h1 className="font-['Space_Grotesk'] text-3xl md:text-4xl font-bold text-white mb-3">
          Stitch hero screens
        </h1>
        <p className="text-[#b9cacb] mb-10 max-w-2xl">
          Tactical layouts from Stitch, wired to live routes. Open a screen, then use Labs for every bootable
          endpoint.
        </p>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {STITCH_HERO_SCREENS.map((screen) => (
            <Link
              key={screen.id}
              to={screen.path}
              className="glass-panel group block rounded-xl border border-white/10 p-5 transition hover:border-[#00dbe9]/50 hover:shadow-[0_0_24px_rgba(0,219,233,0.15)]"
            >
              <span className="material-symbols-outlined text-[#00dbe9] text-3xl mb-3">{screen.icon}</span>
              <h2 className="font-semibold text-lg text-white group-hover:text-[#00dbe9]">{screen.title}</h2>
              <p className="text-sm text-[#849495] mt-1">{screen.subtitle}</p>
              <p className="mt-4 font-mono text-xs text-[#00dbe9]">{screen.path}</p>
            </Link>
          ))}
        </div>
        <div className="mt-10 flex flex-wrap gap-4">
          <Link
            to="/labs"
            className="px-6 py-3 bg-[#00dbe9] text-[#00363a] text-xs font-bold uppercase tracking-wider"
          >
            All labs by field
          </Link>
          <Link to="/labs-fleet" className="px-6 py-3 border border-[#00dbe9]/40 text-[#00dbe9] text-xs font-bold uppercase tracking-wider">
            Live lab fleet
          </Link>
          <Link to="/echo-mission" className="px-6 py-3 border border-[#00dbe9]/40 text-[#00dbe9] text-xs font-bold uppercase tracking-wider">
            Echo mission
          </Link>
          <Link to="/intel-dashboard" className="px-6 py-3 border border-[#00dbe9]/40 text-[#00dbe9] text-xs font-bold uppercase tracking-wider">
            Intel dashboard
          </Link>
          <Link to="/" className="px-6 py-3 border border-[#00dbe9]/40 text-[#00dbe9] text-xs font-bold uppercase tracking-wider">
            Mission control
          </Link>
        </div>
      </main>
    </div>
  );
}
