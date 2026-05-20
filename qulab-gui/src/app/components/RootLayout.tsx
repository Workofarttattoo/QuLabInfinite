import { Outlet, useLocation } from 'react-router';
import { EchoCommandProvider } from '../../lib/echo-command-context';
import { shouldShowEchoCommandBar } from '../../lib/lab-echo-context';
import { EchoLabCommandBar } from './EchoLabCommandBar';

/** Wraps all routes so tactical glass theme CSS is always active. */
export function RootLayout() {
  const { pathname } = useLocation();
  const echoDock = shouldShowEchoCommandBar(pathname);

  return (
    <EchoCommandProvider>
      <div className="qulab-tactical dark min-h-screen min-h-dvh text-foreground antialiased qulab-page-bg">
        <div className={echoDock ? 'pb-48' : undefined}>
          <Outlet />
        </div>
        {echoDock && <EchoLabCommandBar />}
      </div>
    </EchoCommandProvider>
  );
}
